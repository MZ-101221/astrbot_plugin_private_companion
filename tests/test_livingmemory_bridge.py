# -*- coding: utf-8 -*-
"""Contract tests for the in-process LivingMemory bridge.

The bridge is pure glue: it resolves the LivingMemory star, asks its recall
engine for memories and renders a compact bullet list.  These tests pin the
degradation contract (never raise into the reply path) and the rendering rules
without importing AstrBot itself.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import types
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "astrbot_plugin_private_companion"


def _install_import_scaffold() -> None:
    """Expose the bridge module without executing the heavy package inits."""
    packages = (
        (PACKAGE_NAME, PLUGIN_ROOT),
        (f"{PACKAGE_NAME}.companion", PLUGIN_ROOT / "companion"),
        (f"{PACKAGE_NAME}.companion.integrations", PLUGIN_ROOT / "companion" / "integrations"),
    )
    for name, path in packages:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    logging_util = sys.modules.get(f"{PACKAGE_NAME}.logging_util")
    if logging_util is None:
        logging_util = types.ModuleType(f"{PACKAGE_NAME}.logging_util")
        logging_util.get_module_logger = lambda name: logging.getLogger(name)
        sys.modules[f"{PACKAGE_NAME}.logging_util"] = logging_util


_install_import_scaffold()

from astrbot_plugin_private_companion.companion.integrations import (  # noqa: E402
    livingmemory_bridge as bridge,
)


class _Engine:
    def __init__(self, items=None, *, error=None, delay=0.0):
        self.items = items or []
        self.error = error
        self.delay = delay
        self.calls: list[dict] = []

    async def search_memories(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.items


class _Initializer:
    def __init__(self, engine, *, initialized=True):
        self.memory_engine = engine
        self.is_initialized = initialized


class _Star:
    def __init__(self, engine, *, initialized=True):
        self.initializer = _Initializer(engine, initialized=initialized)


class _Metadata:
    def __init__(self, star_cls):
        self.star_cls = star_cls


class _Context:
    def __init__(self, star_cls):
        self._star_cls = star_cls

    def get_registered_star(self, _name):
        return None if self._star_cls is None else _Metadata(self._star_cls)


class _Owner:
    def __init__(self, star_cls=None):
        self.context = _Context(star_cls)


def _item(text="记忆内容", *, create_time=None, score=0.5):
    return {"content": text, "score": score, "metadata": {"create_time": create_time}}


def test_build_viewer_scope_normalises_and_joins():
    assert (
        bridge.build_viewer_scope("aiocqhttp", "MZ-101221")
        == "livingmemory:user:aiocqhttp:mz-101221"
    )
    assert bridge.build_viewer_scope("", "abc") == ""
    assert bridge.build_viewer_scope("qq", "") == ""


def test_render_returns_empty_without_items():
    assert bridge.render([], max_chars=100) == ""
    assert bridge.render([{"content": "   "}], max_chars=100) == ""


def test_render_emits_marker_and_dated_bullets():
    text = bridge.render(
        [
            _item("喜欢深夜改代码", create_time=1780000000.0),
            _item("在改 AstrBot 插件"),
        ],
        max_chars=500,
    )
    lines = text.splitlines()
    assert lines[0] == bridge.BRIDGE_MARKER
    assert "喜欢深夜改代码" in lines[1]
    assert lines[1].startswith("- [")
    assert "在改 AstrBot 插件" in lines[2]
    assert lines[2] == "- 在改 AstrBot 插件"


def test_render_accepts_iso_create_time():
    text = bridge.render([_item("旧格式时间", create_time="2026-06-02T14:19:00+00:00")], max_chars=200)
    assert "2026-06-02" in text


def test_render_truncates_oversized_first_item():
    text = bridge.render([_item("很长的记忆" * 40)], max_chars=120)
    lines = text.splitlines()
    assert len(lines) == 2
    assert lines[1].endswith("…")
    assert len(text) <= 120


def test_render_stops_at_char_budget():
    items = [_item("短记忆" * 5) for _ in range(20)]
    text = bridge.render(items, max_chars=200)
    assert text
    assert len(text) <= 200


def test_resolve_engine_reports_missing_plugin():
    engine, reason = bridge.resolve_engine(_Owner(None))
    assert engine is None
    assert reason == bridge.REASON_PLUGIN_MISSING


def test_resolve_engine_reports_unready_engine():
    engine, reason = bridge.resolve_engine(_Owner(_Star(_Engine(), initialized=False)))
    assert engine is None
    assert reason == bridge.REASON_ENGINE_NOT_READY


def test_resolve_engine_returns_live_engine():
    live = _Engine()
    engine, reason = bridge.resolve_engine(_Owner(_Star(live)))
    assert engine is live
    assert reason == bridge.REASON_OK


def test_recall_ok_path_renders_and_counts():
    live = _Engine([_item("记得你怕吵")])
    result = asyncio.run(
        bridge.recall(
            _Owner(_Star(live)),
            query="最近怎么样",
            top_k=4,
            max_chars=1500,
            timeout_seconds=1.0,
            viewer_scope="livingmemory:user:qq:mz",
            is_group=False,
            allowed_atom_types=None,
        )
    )
    assert result.ok is True
    assert result.reason == bridge.REASON_OK
    assert len(result.items) == 1
    assert bridge.BRIDGE_MARKER in result.text
    sent = live.calls[0]
    assert sent["viewer_scope"] == "livingmemory:user:qq:mz"
    assert sent["is_group"] is False
    assert sent["allowed_atom_types"] is None


def test_recall_passes_group_atom_filter():
    live = _Engine([_item("群里说过的硬件事")])
    asyncio.run(
        bridge.recall(
            _Owner(_Star(live)),
            query="上次说的板子",
            top_k=4,
            max_chars=1500,
            timeout_seconds=1.0,
            is_group=True,
            allowed_atom_types=["episodic", "factual"],
        )
    )
    assert live.calls[0]["allowed_atom_types"] == ["episodic", "factual"]


def test_recall_degrades_on_timeout():
    live = _Engine([_item("慢记忆")], delay=0.5)
    result = asyncio.run(
        bridge.recall(
            _Owner(_Star(live)),
            query="q",
            top_k=4,
            max_chars=500,
            timeout_seconds=0.2,
        )
    )
    assert result.ok is False
    assert result.reason == bridge.REASON_TIMEOUT
    assert result.text == ""


def test_recall_degrades_on_error():
    live = _Engine(error=RuntimeError("boom"))
    result = asyncio.run(
        bridge.recall(
            _Owner(_Star(live)),
            query="q",
            top_k=4,
            max_chars=500,
            timeout_seconds=1.0,
        )
    )
    assert result.ok is False
    assert result.reason == bridge.REASON_ERROR
    assert result.text == ""


def test_recall_degrades_when_plugin_missing():
    result = asyncio.run(
        bridge.recall(
            _Owner(None),
            query="q",
            top_k=4,
            max_chars=500,
            timeout_seconds=1.0,
        )
    )
    assert result.ok is False
    assert result.reason == bridge.REASON_PLUGIN_MISSING


def test_diagnostics_payload_shape():
    result = asyncio.run(
        bridge.recall(
            _Owner(_Star(_Engine([_item("x")]))),
            query="q",
            top_k=4,
            max_chars=500,
            timeout_seconds=1.0,
        )
    )
    payload = result.diagnostics()
    assert set(payload) == {"ok", "reason", "items", "chars", "elapsed_ms", "scope"}
    assert payload["ok"] is True
    assert payload["items"] == 1
    assert isinstance(payload["elapsed_ms"], int)


if __name__ == "__main__":  # pragma: no cover - manual smoke entry
    raise SystemExit(pytest.main([__file__, "-q"]))
