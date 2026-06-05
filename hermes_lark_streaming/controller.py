"""StreamCardController — 流式卡片主控制器（单例）."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import Future as ConcurrentFuture
from typing import Any

from .config import Config
from .feishu import (
    FeishuClient,
    FeishuClientConfig,
)
from .streaming.controller import StreamingController
from .streaming.segments import SegmentType
from .streaming.session import CardSession, SessionState
from .streaming.text import strip_reasoning_tags

_logger = logging.getLogger("hermes_lark_streaming")
_CARD_CREATION_WAIT_SEC = 10.0


class StreamCardController(StreamingController):
    """流式卡片控制器 — 管理多条消息的卡片生命周期."""

    def __init__(self) -> None:
        self._cfg = Config()
        self._client: FeishuClient | None = None
        self._sessions: dict[str, CardSession] = {}
        self._interrupt_map: dict[str, str] = {}
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._session_ttl = self._cfg.card_duration_sec
        self._loop: asyncio.AbstractEventLoop | None = None
        self._text_fallback_needed: set[str] = set()
        self._text_fallback_aliases: dict[str, set[str]] = {}
        # 决策链耗尽/已终态后产生的 background review 消息：session 已无法承载，
        # 按 chat_id 合并后通过 on_background_deliver 包装成独立卡片发送，避免裸文本刷屏。
        self._completed_chat_for_message: dict[str, str] = {}
        self._completed_chat_at: dict[str, float] = {}
        self._orphan_review_queue: dict[str, list[str]] = {}
        self._orphan_review_timers: dict[str, asyncio.TimerHandle] = {}
        self._orphan_review_lock = threading.Lock()
        self._orphan_chat_max_ttl_sec = 600.0  # 已完成 message_id→chat_id 映射的保留窗口
        self._orphan_flush_delay_sec = 0.6  # 同一 chat_id 内的合并窗口

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled and bool(self._cfg.feishu_app_id or self._cfg.env_app_id)

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            app_id = self._cfg.feishu_app_id or self._cfg.env_app_id
            app_secret = self._cfg.feishu_app_secret or self._cfg.env_app_secret
            if not app_id or not app_secret:
                raise RuntimeError("feishu credentials not configured")
            self._client = FeishuClient(
                FeishuClientConfig(
                    app_id=app_id,
                    app_secret=app_secret,
                    base_url=self._cfg.feishu_base_url,
                )
            )
            self._initialized = True

    def _client_ok(self) -> bool:
        return self._initialized and self._client is not None

    def _get_loop(self) -> asyncio.AbstractEventLoop | None:
        """获取事件循环，缓存以便跨线程复用."""
        try:
            loop = asyncio.get_running_loop()
            self._loop = loop
            return loop
        except RuntimeError:
            pass
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        try:
            loop = asyncio.get_event_loop()
            self._loop = loop
            return loop
        except RuntimeError:
            return None

    def _get_active_session(self, message_id: str) -> CardSession | None:
        """获取非终态的活跃 session，不存在或已终态返回 None."""
        session = self._sessions.get(message_id)
        if session is None or session.state.is_terminal:
            return None
        return session

    def _fire_and_forget(
        self,
        coro: Coroutine[Any, Any, Any],
        loop: asyncio.AbstractEventLoop,
    ) -> asyncio.Future[Any] | ConcurrentFuture | None:
        try:
            task = loop.create_task(coro)
            task.add_done_callback(self._on_bg_task_done)
            return task
        except RuntimeError:
            try:
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                fut.add_done_callback(self._on_bg_task_done)
                return fut
            except Exception:
                _logger.debug("fire_and_forget failed", exc_info=True)
                return None

    def on_message_started(
        self,
        *,
        message_id: str | None,
        chat_id: str,
        anchor_id: str | None = None,
    ) -> None:
        """消息处理开始 — 创建会话 + 发占位卡片."""
        if not self.enabled:
            return
        if not message_id:
            _logger.warning("on_message_started: missing message_id, chat=%s", chat_id[:12])
            return
        if message_id in self._sessions:
            return

        self._prune_stale_sessions()

        loop = self._get_loop()
        if loop is None:
            _logger.warning("no event loop available, skipping: msg=%s", message_id[:12])
            return
        session = CardSession(message_id, chat_id, loop)
        self._sessions[message_id] = session
        if anchor_id and anchor_id != message_id:
            session.anchor_id = anchor_id
            self._sessions[anchor_id] = session
        _logger.info("session created: msg=%s chat=%s anchor=%s", message_id[:12], chat_id[:12], (anchor_id or "")[:12])

        session.create_task = self._fire_and_forget(self._do_create_card(session), loop)

    def _mark_text_fallback_needed(self, session: CardSession) -> None:
        keys = {session.message_id}
        if session.anchor_id:
            keys.add(session.anchor_id)
        self._text_fallback_needed.update(keys)
        for key in keys:
            self._text_fallback_aliases[key] = set(keys)

    def consume_text_fallback(self, message_id: str) -> bool:
        """Return whether gateway should undo already_sent and deliver plain text."""
        if message_id not in self._text_fallback_needed:
            return False
        keys = self._text_fallback_aliases.pop(message_id, {message_id})
        for key in keys:
            self._text_fallback_needed.discard(key)
            self._text_fallback_aliases.pop(key, None)
        return True

    def on_thinking(self, *, message_id: str, text: str) -> bool:
        """思考内容增量."""
        if not self.enabled:
            return False
        session = self._get_active_session(message_id)
        if session is None or session.guard.should_skip("on_thinking"):
            return False

        if session.segment_state is None:
            return False
        return self._on_thinking_segment(session, text)

    def on_reasoning(self, *, message_id: str, text: str) -> bool:
        """Native model reasoning delta (incremental append)."""
        if not self.enabled:
            return False
        if not self._cfg.show_reasoning:
            return False
        session = self._get_active_session(message_id)
        if session is None or session.guard.should_skip("on_reasoning"):
            return False

        if session.segment_state is None:
            return False

        session.segment_state.on_reasoning_delta(text)
        self._schedule_flush(session)
        return True

    def on_tool_update(
        self,
        *,
        message_id: str,
        tool_name: str,
        status: str,
        detail: str = "",
    ) -> bool:
        """工具调用事件."""
        if not self.enabled:
            return False
        session = self._get_active_session(message_id)
        if session is None or session.guard.should_skip("on_tool_update"):
            return False
        if session.segment_state is None:
            return False

        if status in ("running", "started", "tool.started"):
            session.tool_use.record_start(tool_name, detail)
        else:
            is_error = status in ("error", "failed")
            session.tool_use.record_end(
                tool_name,
                error=detail if is_error else "",
                output="" if is_error else detail,
            )

        session.segment_state.on_tool_event(len(session.tool_use.build_display_steps()))
        self._schedule_flush(session)
        return True

    def on_answer(self, *, message_id: str, text: str) -> bool:
        """答案文本增量（流式）."""
        if not self.enabled:
            return False
        session = self._get_active_session(message_id)
        if session is None or session.guard.should_skip("on_answer"):
            return False
        if session.segment_state is None:
            return False

        answer_text = strip_reasoning_tags(text)
        if not answer_text:
            return False

        session.segment_state.on_answer_delta(answer_text)
        self._schedule_flush(session)
        return True

    def on_aborted(self, *, message_id: str) -> None:
        """用户 /stop 导致消息被中断."""
        if not self.enabled:
            return
        session = self._get_active_session(message_id)
        if session is None:
            return

        session.state = SessionState.ABORTED
        session.flush.mark_completed()
        _logger.info("on_aborted: msg=%s state=ABORTED", message_id[:12])

        self._complete_session(session)

    def on_interrupted(
        self,
        *,
        old_message_id: str,
        new_message_id: str,
        chat_id: str,
        anchor_id: str | None = None,
    ) -> None:
        """用户发送新消息导致前一条消息被中断 — abort A + create B."""
        if not self.enabled:
            return

        old_session = self._get_active_session(old_message_id)
        if old_session is not None:
            old_session.state = SessionState.ABORTED
            old_session.flush.mark_completed()
            _logger.info(
                "on_interrupted: abort old msg=%s",
                old_message_id[:12],
            )
            self._complete_session(old_session)

        if new_message_id not in self._sessions:
            loop = self._get_loop()
            if loop is not None:
                reply_anchor_id = anchor_id if anchor_id and anchor_id != new_message_id else None
                session = CardSession(new_message_id, chat_id, loop)
                session.anchor_id = reply_anchor_id
                self._sessions[new_message_id] = session
                if reply_anchor_id:
                    self._sessions[reply_anchor_id] = session
                _logger.info(
                    "on_interrupted: create new msg=%s chat=%s anchor=%s",
                    new_message_id[:12],
                    chat_id[:12],
                    (reply_anchor_id or new_message_id)[:12],
                )
                session.create_task = self._fire_and_forget(self._do_create_card(session), loop)

        self._interrupt_map[old_message_id] = new_message_id
        for key, val in list(self._interrupt_map.items()):
            if val == old_message_id:
                self._interrupt_map[key] = new_message_id

    def on_completed(
        self,
        *,
        message_id: str,
        answer: str = "",
        duration: float = 0.0,
        model: str = "",
        tokens: dict | None = None,
        context: dict | None = None,
    ) -> bool:
        """消息处理完成 — 构建终端卡片."""
        if not self.enabled:
            return False
        session = self._completion_session(message_id)
        if session is None:
            return False
        message_id = session.message_id

        # 卡片创建失败 → 交回 gateway 正常回复
        if session.state == SessionState.FAILED:
            _logger.info("on_completed: msg=%s state=FAILED, yielding to gateway", message_id[:12])
            self._mark_text_fallback_needed(session)
            self._cleanup(message_id)
            return False

        _logger.info(
            "on_completed: msg=%s has_card=%s state=%s",
            message_id[:12],
            session.has_card,
            session.state,
        )

        self._apply_completion_payload(
            session=session,
            answer=answer,
            duration=duration,
            model=model,
            tokens=tokens,
            context=context,
        )

        self._complete_session(session)
        return True

    async def on_completed_wait(
        self,
        *,
        message_id: str,
        answer: str = "",
        duration: float = 0.0,
        model: str = "",
        tokens: dict | None = None,
        context: dict | None = None,
    ) -> bool:
        """消息处理完成，并等待卡片真正收尾后返回是否已发送."""
        if not self.enabled:
            return False
        session = self._completion_session(message_id)
        if session is None:
            return False
        message_id = session.message_id

        if not await self._wait_for_card_creation(session):
            _logger.info("on_completed_wait: msg=%s card creation not ready, yielding to gateway", message_id[:12])
            self._mark_text_fallback_needed(session)
            self._cleanup(message_id)
            return False

        if session.state == SessionState.FAILED:
            _logger.info("on_completed_wait: msg=%s state=FAILED, yielding to gateway", message_id[:12])
            self._mark_text_fallback_needed(session)
            self._cleanup(message_id)
            return False

        if not session.has_card:
            _logger.info("on_completed_wait: msg=%s has no card, yielding to gateway", message_id[:12])
            self._mark_text_fallback_needed(session)
            self._cleanup(message_id)
            return False

        _logger.info(
            "on_completed_wait: msg=%s has_card=%s state=%s",
            message_id[:12],
            session.has_card,
            session.state,
        )

        # 触顶场景: session 即将进入终态,但 hermes 不会自动 schedule background
        # review(因为 nudge_interval=10 不满足)。主动入队一条 review 走合并卡片
        # 路径,确保"决策链耗尽"事件以独立卡片形式呈现给用户,而不是裸文本警告。
        self._enqueue_budget_exhausted_review(
            session=session,
            answer=answer,
            context=context,
        )

        self._apply_completion_payload(
            session=session,
            answer=answer,
            duration=duration,
            model=model,
            tokens=tokens,
            context=context,
        )

        return await self._complete_session_wait(session)

    async def try_finalize_by_chat(
        self,
        *,
        chat_id: str,
        answer: str = "",
        duration: float = 0.0,
        model: str = "",
        tokens: dict | None = None,
        context: dict | None = None,
    ) -> bool:
        """Fallback 路径: on_completed_wait 失败后, 找同 chat 最近可 finalize 的 session.

        场景: event.message_id 无法定位 session（compaction 切走 / cleanup 已删 / 并发争用）,
        但同 chat 有 _已成功发占位卡_ 的 session（_prune 之前）, 用 cardkit_update 追加 answer。

        成功返回 True（已 patch 到原卡）；失败返回 False（gateway 走 text fallback 发 post）。
        """
        if not self.enabled or not chat_id:
            return False
        # 找同 chat 最近且 _有 card_id_ 的 session（说明占位卡已发成功）
        candidates = [
            s for s in self._sessions.values()
            if s.chat_id == chat_id
            and s.card_id
            and not s.state.is_terminal
        ]
        if not candidates:
            return False
        # 选最近创建的
        session = max(candidates, key=lambda s: s.created_at)
        # 调 _complete_session_wait（它内部调 _do_complete_card）
        if answer and session.segment_state and not any(
            seg.type == SegmentType.ANSWER for seg in session.segment_state.segments
        ):
            final_answer = strip_reasoning_tags(answer)
            if final_answer:
                session.segment_state.on_answer_delta(final_answer)
        session.footer = {
            "duration": duration,
            "model": model,
            **({"input_tokens": tokens.get("input_tokens")} if tokens else {}),
            **({"output_tokens": tokens.get("output_tokens")} if tokens else {}),
            **({"context_used": context.get("used_tokens")} if context else {}),
            **({"context_max": context.get("max_tokens")} if context else {}),
        }
        return await self._complete_session_wait(session)

    async def try_finalize_all_by_chat(
        self,
        *,
        chat_id: str,
    ) -> int:
        """Turn end 兑底: 找同 chat 所有 _非_ terminal session, 全部 finalize.

        场景: on_completed_wait 失败 + try_finalize_by_chat 仍失败时, 表明
        answer 所在 session 找不到且 _chat 范围内_ 也没最近的可用 session。
        此时仍有 _同 chat 但未 finalized_ 的早期 session(被多次切走/过其窗口)
        遺留在 _sessions dict,需兑底全部加 footer,避免用户看到"半成品卡堆积"。

        返回成功 finalize 的 session 数。
        """
        if not self.enabled or not chat_id:
            return 0
        candidates = [
            s for s in self._sessions.values()
            if s.chat_id == chat_id
            and s.card_id
            and not s.state.is_terminal
        ]
        success = 0
        for session in candidates:
            if not session.footer:
                session.footer = {
                    "duration": 0.0,
                    "model": "completed-no-footer",
                    "completed_no_footer": True,
                }
            ok = await self._complete_session_wait(session)
            if ok:
                success += 1
        return success

    def on_cron_deliver(
        self,
        *,
        chat_id: str,
        content: str,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        """Cron 推送 — 包装为静态卡片发送，成功返回 True."""
        if not self.enabled or not content or not chat_id:
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._do_cron_deliver(chat_id, content), loop
        )
        try:
            future.result(timeout=30)
            _logger.info("cron card delivered: chat=%s len=%d", chat_id[:12], len(content))
            return True
        except Exception:
            _logger.warning("cron card delivery failed", exc_info=True)
            return False

    def on_status_message(
        self,
        *,
        chat_id: str,
        event_type: str,
        content: str,
        metadata: dict | None = None,
    ) -> bool:
        """状态消息 (compaction/触顶/系统通知) — 推独立卡片.

        对于 Compacting context / Iteration budget exhausted 类,会先强制
        finalize 当前 chat 的 active session,让"截断自动整合面板"发生。
        同步包装:调度到 gateway event loop,等 30s。
        """
        if not self.enabled or not chat_id or not content:
            return False
        loop = self._get_loop()
        if loop is None:
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._do_status_message(
                chat_id=chat_id,
                event_type=event_type,
                content=content,
                metadata=metadata,
            ),
            loop,
        )
        try:
            return bool(future.result(timeout=30))
        except Exception:
            _logger.warning("on_status_message failed: chat=%s type=%s",
                            chat_id[:12], event_type, exc_info=True)
            return False

    async def _do_status_message(
        self,
        *,
        chat_id: str,
        event_type: str,
        content: str,
        metadata: dict | None = None,
    ) -> bool:
        """async 实际处理:compaction/budget 强制 finalize active session + 推独立卡片."""
        is_compaction = "Compacting context" in content
        is_budget = "Iteration budget exhausted" in content or content.startswith("⚠️")

        # 截断事件:强制 finalize 当前 chat 的 active streaming 卡片
        if is_compaction or is_budget:
            active_sessions = [
                s for s in self._sessions.values()
                if s.chat_id == chat_id
                and s.state in (SessionState.STREAMING, SessionState.ACTIVE)
            ]
            for session in active_sessions:
                try:
                    await self._do_complete_card(session)
                    _logger.info(
                        "on_status_message: finalized active session msg=%s due to %s",
                        session.message_id[:12],
                        "compaction" if is_compaction else "budget",
                    )
                except Exception as exc:
                    _logger.warning(
                        "on_status_message: finalize failed msg=%s: %s",
                        session.message_id[:12], exc,
                    )

        # 发独立 status 卡片
        try:
            return await self._do_send_standalone_status_card(
                chat_id=chat_id,
                event_type=event_type,
                content=content,
                metadata=metadata,
            )
        except Exception as exc:
            _logger.warning("on_status_message: standalone card failed: %s", exc, exc_info=True)
            return False

    async def _do_send_standalone_status_card(
        self,
        *,
        chat_id: str,
        event_type: str,
        content: str,
        metadata: dict | None = None,
    ) -> bool:
        """推独立 status 卡片(不 reply to user message)."""
        from .cardkit.builder import build_status_card
        await self._ensure_init()
        assert self._client is not None
        card = build_status_card(event_type=event_type, content=content)
        reply_to = (metadata or {}).get("reply_to_message_id")
        await self._client.send_card_to_chat(
            chat_id,
            card,
            reply_to_message_id=reply_to,
        )
        _logger.info("on_status_message: standalone card sent chat=%s type=%s len=%d",
                     chat_id[:12], event_type, len(content))
        return True

    async def on_background_deliver(
        self,
        *,
        chat_id: str,
        preview: str,
        content: str,
        reply_to_message_id: str | None = None,
    ) -> bool:
        """Background 任务完成推送 — 包装为静态卡片发送，成功返回 True."""
        if not self.enabled or not content or not chat_id:
            return False
        try:
            await self._do_background_deliver(
                chat_id,
                preview,
                content,
                reply_to_message_id=reply_to_message_id,
            )
            _logger.info("background card delivered: chat=%s len=%d", chat_id[:12], len(content))
            return True
        except Exception:
            _logger.warning("background card delivery failed", exc_info=True)
            return False

    def defer_background_review(
        self,
        *,
        message_id: str,
        text: str,
        sender: Callable[[str], Any],
    ) -> bool:
        """暂存 Hermes background review 通知，等卡片收尾后再发送.

        两条路径：
        1. session 仍活跃 → 暂存到 session.deferred_background_reviews，
           由 _flush_deferred_background_reviews 在卡片收尾时按 sender 发送。
        2. session 不存在/已终态（典型：决策链耗尽 90/90 触发的 AI 总结，
           agent 完成后 callback 又追加了 memory/cost/compaction 等 review
           通知）→ 查 _completed_chat_for_message 拿到 chat_id，
           把消息加入 _orphan_review_queue（按 chat_id 合并），
           schedule 短延迟后调 on_background_deliver 包装成独立卡片推出去，
           避免 fallback 走裸文本刷屏。
        """
        if not self.enabled or not text or not callable(sender):
            return False
        session = self._get_active_session(message_id)
        if session is not None:
            with session.deferred_background_review_lock:
                if session.deferred_background_review_closed:
                    return False
                session.deferred_background_reviews.append((text, sender))
            return True
        # 孤儿分支：session 已不可用。
        chat_id = self._completed_chat_for_message.get(message_id)
        if not chat_id:
            return False
        self._prune_completed_chat_map()
        return self._enqueue_orphan_review(chat_id, text)

    def _enqueue_orphan_review(self, chat_id: str, text: str) -> bool:
        """孤儿 review 入队 + schedule 合并 flush."""
        with self._orphan_review_lock:
            queue = self._orphan_review_queue.setdefault(chat_id, [])
            queue.append(text)
            existing = self._orphan_review_timers.pop(chat_id, None)
            if existing is not None:
                try:
                    existing.cancel()
                except Exception:
                    pass
            loop = self._get_loop()
            if loop is None or loop.is_closed():
                # 无 event loop 可调度 — 同步 flush 兜底。
                return self._flush_orphan_reviews_sync(chat_id)
            try:
                handle = loop.call_later(
                    self._orphan_flush_delay_sec,
                    self._flush_orphan_reviews_sync,
                    chat_id,
                )
                self._orphan_review_timers[chat_id] = handle
            except RuntimeError:
                return self._flush_orphan_reviews_sync(chat_id)
        return True

    def _enqueue_budget_exhausted_review(
        self,
        *,
        session: Any,
        answer: str,
        context: dict | None,
    ) -> bool:
        """触顶场景:主动入队一条 synthetic review 走孤儿合并路径.

        触发条件:context.turn_exit_reason 以 'max_iterations_reached' 开头。
        此时 hermes 不会 schedule background review(因为 _should_review_memory
        仍是 False),所以孤儿 review 队列永远不会被填充 → 用户只看到
        'Iteration budget exhausted' 裸文本警告,没有任何事后提示。

        主动入队后:
        - 如果 hermes 后续 schedule memory review(10+ turns 时): 我们的 synthetic
          review 会和真 review 合并,走 on_background_deliver 推一张合并卡片。
        - 如果 hermes 不 schedule: 只有这条 synthetic review,也会走合并路径
          推一张单条卡片(同样避免裸文本)。

        必须在 session._cleanup 之前调用,因为依赖 session.chat_id。
        """
        if not self.enabled:
            return False
        if not context:
            return False
        turn_exit_reason = context.get("turn_exit_reason", "") or ""
        if not turn_exit_reason.startswith("max_iterations_reached"):
            return False
        chat_id = getattr(session, "chat_id", None)
        if not chat_id:
            return False
        api_calls = context.get("api_calls", "?")
        max_iter = context.get("max_iterations", "?")
        text = (
            f"⚠️ Decision chain exhausted at {api_calls}/{max_iter} iterations.\n"
            f"Summary delivered: {len(answer or '')} chars.\n"
            f"Background review skipped — budget hit before review window."
        )
        ok = self._enqueue_orphan_review(chat_id, text)
        if ok:
            _logger.info(
                "budget-exhausted review enqueued: chat=%s exit=%s",
                chat_id[:12], turn_exit_reason,
            )
        return ok

    def _flush_orphan_reviews_sync(self, chat_id: str) -> bool:
        """同步入口：取 queue、清理 timer、起 async flush 任务."""
        with self._orphan_review_lock:
            self._orphan_review_timers.pop(chat_id, None)
            pending = self._orphan_review_queue.pop(chat_id, [])
        if not pending:
            return True
        loop = self._get_loop()
        if loop is None or loop.is_closed():
            _logger.warning(
                "orphan reviews dropped: no event loop, chat=%s count=%d",
                chat_id[:12], len(pending),
            )
            return False
        try:
            task = loop.create_task(self._do_flush_orphan_reviews(chat_id, pending))
            task.add_done_callback(self._on_bg_task_done)
        except RuntimeError:
            _logger.warning(
                "orphan reviews dropped: create_task failed, chat=%s count=%d",
                chat_id[:12], len(pending),
            )
            return False
        return True

    async def _do_flush_orphan_reviews(self, chat_id: str, items: list[str]) -> None:
        """合并孤儿 review，调 on_background_deliver 推独立卡片."""
        if not self.enabled:
            return
        content = "\n".join(items).strip()
        if not content or not chat_id:
            return
        preview = f"(decision chain exhausted · {len(items)} follow-up note{'s' if len(items) != 1 else ''})"
        try:
            await self._do_background_deliver(chat_id, preview, content)
            _logger.info(
                "orphan reviews flushed: chat=%s count=%d len=%d",
                chat_id[:12], len(items), len(content),
            )
        except Exception:
            _logger.warning(
                "orphan reviews flush failed: chat=%s count=%d",
                chat_id[:12], len(items), exc_info=True,
            )

    def _prune_completed_chat_map(self) -> None:
        """清理过期的 message_id→chat_id 映射，避免内存泄漏."""
        if not self._completed_chat_at:
            return
        now = time.time()
        expired = [
            k for k, t in self._completed_chat_at.items()
            if now - t > self._orphan_chat_max_ttl_sec
        ]
        for k in expired:
            self._completed_chat_at.pop(k, None)
            self._completed_chat_for_message.pop(k, None)

    def _flush_deferred_background_reviews(self, session: CardSession) -> None:
        lock = getattr(session, "deferred_background_review_lock", None)
        reviews = getattr(session, "deferred_background_reviews", None)
        if lock is None or reviews is None:
            return
        with lock:
            session.deferred_background_review_closed = True
            pending = list(reviews)
            reviews.clear()
        for text, sender in pending:
            try:
                sender(text)
            except Exception:
                _logger.debug("background review sender failed", exc_info=True)

    def _cleanup(self, message_id: str) -> None:
        session = self._sessions.pop(message_id, None)
        if session is None:
            return
        anchor = getattr(session, "anchor_id", None)
        chat_id = getattr(session, "chat_id", None)
        if chat_id:
            # 保留 message_id→chat_id 映射，供 defer_background_review 在孤儿分支
            # 复用（决策链耗尽/已终态的 session 已被清掉，但 background review
            # 消息仍可能产生）。
            now = time.time()
            for key in (message_id, anchor):
                if key and key not in self._completed_chat_for_message:
                    self._completed_chat_for_message[key] = chat_id
                    self._completed_chat_at[key] = now
        if anchor and self._sessions.get(anchor) is session:
            del self._sessions[anchor]
        stale_keys = [k for k, v in self._interrupt_map.items() if v == message_id]
        for k in stale_keys:
            del self._interrupt_map[k]
        session.flush.mark_completed()
        if session.image_resolver:
            session.image_resolver.cancel_pending()

    def _completion_session(self, message_id: str) -> CardSession | None:
        session = self._sessions.get(message_id)
        if session is not None and (not session.state.is_terminal or session.state == SessionState.FAILED):
            return session

        redirected_id = self._interrupt_map.pop(message_id, None)
        if redirected_id is not None:
            _logger.info(
                "on_completed: redirect msg=%s -> msg=%s",
                message_id[:12],
                redirected_id[:12],
            )
            redirected = self._sessions.get(redirected_id)
            if redirected is not None and not redirected.state.is_terminal:
                return redirected
        return None

    async def _wait_for_card_creation(self, session: CardSession) -> bool:
        task = session.create_task
        if task is None:
            return True
        try:
            if isinstance(task, asyncio.Future):
                await asyncio.wait_for(task, timeout=_CARD_CREATION_WAIT_SEC)
            else:
                await asyncio.wait_for(asyncio.wrap_future(task), timeout=_CARD_CREATION_WAIT_SEC)
            return True
        except TimeoutError:
            _logger.warning(
                "card creation timed out: msg=%s timeout=%.1fs",
                session.message_id[:12],
                _CARD_CREATION_WAIT_SEC,
            )
            task.cancel()
            session.mark_failed()
            return False
        except asyncio.CancelledError:
            session.mark_failed()
            return False
        except Exception:
            _logger.debug("card creation task failed", exc_info=True)
            return False

    def _apply_completion_payload(
        self,
        *,
        session: CardSession,
        answer: str,
        duration: float,
        model: str,
        tokens: dict | None,
        context: dict | None,
    ) -> None:
        if answer and session.segment_state and not any(
            seg.type == SegmentType.ANSWER for seg in session.segment_state.segments
        ):
            final_answer = strip_reasoning_tags(answer)
            if final_answer:
                session.segment_state.on_answer_delta(final_answer)

        session.footer = {
            "duration": duration,
            "model": model,
            **({"input_tokens": tokens.get("input_tokens")} if tokens else {}),
            **({"output_tokens": tokens.get("output_tokens")} if tokens else {}),
            **({"context_used": context.get("used_tokens")} if context else {}),
            **({"context_max": context.get("max_tokens")} if context else {}),
        }

    def _complete_session(self, session: CardSession) -> None:
        """异步完成当前流式卡片."""
        session.flush.mark_completed()
        self._fire_and_forget(self._do_complete_card(session), session._loop)

    async def _complete_session_wait(self, session: CardSession) -> bool:
        """完成当前流式卡片，并等待最终 API 结果."""
        session.flush.mark_completed()
        return await self._do_complete_card(session)

    def _prune_stale_sessions(self) -> None:
        now = time.time()
        stale = [mid for mid, s in self._sessions.items() if mid is not None and now - s.created_at > self._session_ttl]
        for mid in stale:
            _logger.warning("pruning stale session: msg=%s", mid[:12])
            self._cleanup(mid)

    @staticmethod
    def _on_bg_task_done(fut: asyncio.Future[Any] | ConcurrentFuture) -> None:
        try:
            fut.result()
        except asyncio.CancelledError:
            return
        except Exception:
            _logger.warning("background task failed", exc_info=True)


_controller: StreamCardController | None = None
_controller_lock = threading.Lock()


def get_controller() -> StreamCardController:
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = StreamCardController()
        return _controller
