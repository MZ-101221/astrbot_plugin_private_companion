# -*- coding: utf-8 -*-
"""In-process bridge to the LivingMemory plugin's recall engine.

LivingMemory owns long-term memory; this module lets the companion ask it for
relevant memories without importing the plugin, opening a port, or depending on
its HTTP API.  Every failure degrades to an empty result carrying a stable
reason code so a missing or half-initialised optional plugin can never break the
reply path.

The bridge deliberately renders its own compact bullet list instead of reusing
LivingMemory's ``format_memories_for_injection``: that helper wraps its output
in ``<RAG-Faiss-Memory>`` markers, and nesting those inside a companion prompt
section makes traces hard to read.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ...logging_util import get_module_logger

logger = get_module_logger(__name__)

STAR_NAME = "astrbot_plugin_livingmemory"
USER_SCOPE_PREFIX = "livingmemory:user:"
BRIDGE_MARKER = "<!-- lm-bridge:v1 -->"

REASON_OK = "ok"
REASON_PLUGIN_MISSING = "plugin_missing"
REASON_ENGINE_NOT_READY = "engine_not_ready"
REASON_TIMEOUT = "timeout"
REASON_ERROR = "error"

_MAX_ITEM_CHARS = 240
_MIN_ITEM_CHARS = 24


@dataclass(frozen=True, slots=True)
class LivingMemoryRecall:
    """Outcome of one bridge recall attempt."""

    ok: bool
    reason: str
    items: tuple[dict[str, Any], ...] = ()
    text: str = ""
    elapsed_ms: int = 0
    scope: str = ""

    def diagnostics(self) -> dict[str, Any]:
        """Return the payload exposed on the diagnostics endpoint."""
        return {
            "ok": self.ok,
            "reason": self.reason,
            "items": len(self.items),
            "chars": len(self.text),
            "elapsed_ms": self.elapsed_ms,
            "scope": self.scope,
        }


def _star_metadata(owner: Any) -> Any | None:
    """Return LivingMemory's registry entry, or ``None`` when it is not loaded."""
    context = getattr(owner, "context", None)
    getter = getattr(context, "get_registered_star", None)
    if callable(getter):
        try:
            metadata = getter(STAR_NAME)
        except Exception:
            metadata = None
        if metadata is not None:
            return metadata
    get_all = getattr(context, "get_all_stars", None)
    if not callable(get_all):
        return None
    try:
        stars = list(get_all() or [])
    except Exception:
        return None
    for metadata in stars:
        names = (
            getattr(metadata, "name", ""),
            getattr(metadata, "root_dir_name", ""),
            getattr(metadata, "module_path", ""),
        )
        if any(STAR_NAME in str(name or "") for name in names):
            return metadata
    return None


def resolve_engine(owner: Any) -> tuple[Any | None, str]:
    """Resolve LivingMemory's memory engine without importing the plugin.

    Returns:
        ``(engine, reason)``.  ``reason`` is :data:`REASON_OK` only when the
        engine is present and the plugin reports itself initialised.
    """
    metadata = _star_metadata(owner)
    if metadata is None:
        return None, REASON_PLUGIN_MISSING
    plugin = getattr(metadata, "star_cls", None)
    initializer = getattr(plugin, "initializer", None)
    if initializer is None:
        return None, REASON_ENGINE_NOT_READY
    if not bool(getattr(initializer, "is_initialized", False)):
        return None, REASON_ENGINE_NOT_READY
    engine = getattr(initializer, "memory_engine", None)
    if engine is None or not callable(getattr(engine, "search_memories", None)):
        return None, REASON_ENGINE_NOT_READY
    return engine, REASON_OK


def build_viewer_scope(platform: Any, subject: Any) -> str:
    """Map a platform identity onto LivingMemory's user-scope string.

    LivingMemory isolates per-user memories under
    ``livingmemory:user:<platform>:<identity>``.  Both segments are casefolded
    so the same account keeps one scope across adapters.
    """
    platform_text = str(platform or "").strip().casefold()
    subject_text = str(subject or "").strip().casefold()
    if not platform_text or not subject_text:
        return ""
    return f"{USER_SCOPE_PREFIX}{platform_text}:{subject_text}"


def _single_line(value: Any, limit: int = _MAX_ITEM_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _format_day(metadata: dict[str, Any]) -> str:
    raw = metadata.get("create_time")
    stamp: float | None = None
    try:
        stamp = float(raw)
    except (TypeError, ValueError):
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
            except ValueError:
                return ""
            return parsed.strftime("%Y-%m-%d")
        return ""
    if stamp is None or stamp <= 0:
        return ""
    try:
        return datetime.fromtimestamp(stamp).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return ""


def _item_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("content") or "")
    return str(getattr(item, "content", "") or "")


def _item_metadata(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        raw = item.get("metadata")
    else:
        raw = getattr(item, "metadata", None)
    return raw if isinstance(raw, dict) else {}


def render(items: Any, *, max_chars: int) -> str:
    """Render recall items as a compact bullet list bounded by ``max_chars``."""
    if max_chars <= 0:
        return ""
    lines: list[str] = [BRIDGE_MARKER]
    used = len(BRIDGE_MARKER)
    for item in items or ():
        body = _single_line(_item_text(item))
        if not body:
            continue
        day = _format_day(_item_metadata(item))
        prefix = f"- [{day}] " if day else "- "
        remaining = max_chars - used - 1 - len(prefix)
        if remaining <= 0:
            break
        if len(body) > remaining:
            # A single oversized first item would otherwise render nothing at
            # all; keep a truncated bullet while the budget still fits one.
            if len(lines) > 1 or remaining < _MIN_ITEM_CHARS:
                break
            body = _single_line(body, remaining)
        lines.append(prefix + body)
        used += 1 + len(prefix) + len(body)
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


async def recall(
    owner: Any,
    *,
    query: str,
    top_k: int,
    max_chars: int,
    timeout_seconds: float,
    session_id: str | None = None,
    persona_id: str | None = None,
    extra_scopes: list[str] | None = None,
    viewer_scope: str | None = None,
    is_group: bool = False,
    allowed_atom_types: list[str] | None = None,
) -> LivingMemoryRecall:
    """Ask LivingMemory for memories relevant to ``query``.

    Args:
        owner: Companion plugin instance (needs ``context``).
        query: Search text, normally the current inbound message.
        top_k: Maximum number of memories to request.
        max_chars: Character budget for the rendered block.
        timeout_seconds: Hard deadline for the whole call.
        session_id: Current ``unified_msg_origin`` scope.
        persona_id: Active persona id.
        extra_scopes: Additional scopes such as channel memory.
        viewer_scope: ``livingmemory:user:<platform>:<identity>`` of the speaker.
        is_group: Whether the event is a group message.
        allowed_atom_types: Restrict memory types (group chat uses
            ``["episodic", "factual"]``); ``None`` disables type filtering.

    Returns:
        :class:`LivingMemoryRecall`; ``ok`` is false on every degraded path.
    """
    engine, reason = resolve_engine(owner)
    if engine is None:
        return LivingMemoryRecall(ok=False, reason=reason, scope=viewer_scope or "")

    started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "query": query,
        "k": max(1, min(10, int(top_k))),
        "session_id": session_id,
        "persona_id": persona_id,
        "extra_scopes": extra_scopes or None,
        "viewer_scope": viewer_scope or None,
        "is_group": bool(is_group),
        "allowed_atom_types": allowed_atom_types,
    }
    try:
        items = await asyncio.wait_for(
            engine.search_memories(**kwargs),
            timeout=max(0.2, float(timeout_seconds)),
        )
    except asyncio.TimeoutError:
        logger.debug("LivingMemory 桥接超时（%s 秒）", timeout_seconds)
        return LivingMemoryRecall(
            ok=False,
            reason=REASON_TIMEOUT,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            scope=viewer_scope or "",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # optional-plugin boundary: mapped to a stable reason
        logger.warning("LivingMemory 桥接失败：%s: %s", type(exc).__name__, exc)
        return LivingMemoryRecall(
            ok=False,
            reason=REASON_ERROR,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            scope=viewer_scope or "",
        )

    text = render(items, max_chars=max_chars)
    return LivingMemoryRecall(
        ok=bool(text),
        reason=REASON_OK,
        items=tuple(items or ()),
        text=text,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        scope=viewer_scope or "",
    )


__all__ = [
    "BRIDGE_MARKER",
    "LivingMemoryRecall",
    "REASON_ENGINE_NOT_READY",
    "REASON_ERROR",
    "REASON_OK",
    "REASON_PLUGIN_MISSING",
    "REASON_TIMEOUT",
    "STAR_NAME",
    "USER_SCOPE_PREFIX",
    "build_viewer_scope",
    "recall",
    "render",
    "resolve_engine",
]
