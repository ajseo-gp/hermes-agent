"""Tests for the generic ``pre_exec`` plugin hook and its terminal fire site.

The hook is an opt-in extension of the terminal pre-execution guard chain
(``tools.terminal_tool._pre_exec_block``). It is deliberately generic: the
resolver in ``tools.pre_exec_hook`` knows nothing about the terminal wire
format, and the terminal owns rendering the refusal envelope.

Covers the four verdict paths plus the absent case:

1. Absent   — no callback registered: ``invoke_hook`` is never called and the
   guard chain behaves exactly as it did before the hook existed.
2. Allow    — ``{"action": "allow"}`` proceeds, and is advisory only: it does
   not override a built-in guard, nor veto another plugin's block.
3. Block    — ``{"action": "block", "reason": ...}`` refuses the command with
   that reason.
4. Non-applicable — ``None`` / non-dict / unknown action / block without a
   reason: ignored, the command proceeds.
5. Fallback — a callback that raises is isolated by ``invoke_hook`` and the
   command proceeds (fail open), identical to the absent case.

Mirrors the hook-test conventions in ``tests/tools/test_pre_transcription_hook.py``.
"""

from __future__ import annotations

import json

import pytest

import hermes_cli.plugins as plugins_mod
from tools import pre_exec_hook
from tools.pre_exec_hook import PRE_EXEC_HOOK, PreExecDecision, resolve_pre_exec


CONTEXT = {
    "command": "echo hello",
    "env_type": "local",
    "cwd": "/tmp/project",
    "workdir": None,
    "session_key": "session-1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_hooks(monkeypatch, results):
    """Install fake has_hook/invoke_hook returning *results* and capture kwargs."""
    captured = {}

    def _invoke(hook_name, **kw):
        captured["hook_name"] = hook_name
        captured["kwargs"] = kw
        return list(results)

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke)
    return captured


def _no_hooks(monkeypatch):
    """No hook registered: has_hook is False and invoke_hook must not fire."""
    def _boom(hook_name, **kw):  # pragma: no cover - the assert is the point
        raise AssertionError("invoke_hook must not be called when has_hook() is False")

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _boom)


# ---------------------------------------------------------------------------
# Registration surface
# ---------------------------------------------------------------------------


def test_pre_exec_in_valid_hooks():
    assert PRE_EXEC_HOOK == "pre_exec"
    assert PRE_EXEC_HOOK in plugins_mod.VALID_HOOKS


def test_pre_exec_is_python_plugin_only():
    """The shell-hook response parser has no channel for this directive, so a
    shell registration must be refused loudly rather than silently ignored."""
    assert PRE_EXEC_HOOK in plugins_mod.SHELL_UNSUPPORTED_HOOKS


# ---------------------------------------------------------------------------
# 1. Absent
# ---------------------------------------------------------------------------


class TestHookAbsent:
    def test_resolver_short_circuits_without_invoking(self, monkeypatch):
        _no_hooks(monkeypatch)
        assert resolve_pre_exec(**CONTEXT) == PreExecDecision(blocked=False, reason=None)

    def test_decision_is_not_blocked(self, monkeypatch):
        _no_hooks(monkeypatch)
        assert resolve_pre_exec(**CONTEXT).blocked is False

    def test_registered_but_returning_nothing_also_proceeds(self, monkeypatch):
        _fake_hooks(monkeypatch, [])
        assert resolve_pre_exec(**CONTEXT).blocked is False


# ---------------------------------------------------------------------------
# 2. Allow
# ---------------------------------------------------------------------------


class TestAllow:
    def test_allow_proceeds(self, monkeypatch):
        _fake_hooks(monkeypatch, [{"action": "allow"}])
        assert resolve_pre_exec(**CONTEXT).blocked is False

    def test_allow_does_not_veto_another_plugins_block(self, monkeypatch):
        """Allow is advisory. A block from any callback wins regardless of order."""
        _fake_hooks(monkeypatch, [{"action": "allow"}, {"action": "block", "reason": "no"}])
        decision = resolve_pre_exec(**CONTEXT)
        assert decision.blocked is True
        assert decision.reason == "no"


# ---------------------------------------------------------------------------
# 3. Block
# ---------------------------------------------------------------------------


class TestBlock:
    def test_block_with_reason(self, monkeypatch):
        _fake_hooks(monkeypatch, [{"action": "block", "reason": "guard says no"}])
        decision = resolve_pre_exec(**CONTEXT)
        assert decision.blocked is True
        assert decision.reason == "guard says no"

    def test_first_block_reason_wins(self, monkeypatch):
        _fake_hooks(monkeypatch, [
            {"action": "block", "reason": "first"},
            {"action": "block", "reason": "second"},
        ])
        assert resolve_pre_exec(**CONTEXT).reason == "first"

    def test_reason_is_truncated(self, monkeypatch):
        _fake_hooks(monkeypatch, [{"action": "block", "reason": "x" * 10_000}])
        reason = resolve_pre_exec(**CONTEXT).reason
        assert len(reason) <= pre_exec_hook.MAX_REASON_CHARS


# ---------------------------------------------------------------------------
# 4. Non-applicable
# ---------------------------------------------------------------------------


class TestNonApplicable:
    @pytest.mark.parametrize("result", [
        None,
        {},
        "block",
        ["block"],
        42,
        {"action": "modify"},
        {"action": "deny", "reason": "wrong verb"},
        {"reason": "no action key"},
        {"action": "block"},               # a block MUST carry a reason
        {"action": "block", "reason": ""},
        {"action": "block", "reason": "   "},
        {"action": "block", "reason": 7},
    ])
    def test_ignored_and_command_proceeds(self, monkeypatch, result):
        _fake_hooks(monkeypatch, [result])
        assert resolve_pre_exec(**CONTEXT).blocked is False

    def test_non_applicable_does_not_mask_a_later_block(self, monkeypatch):
        _fake_hooks(monkeypatch, [None, {"action": "block", "reason": "late"}])
        assert resolve_pre_exec(**CONTEXT).reason == "late"


# ---------------------------------------------------------------------------
# 5. Fallback — a raising callback degrades to absent
# ---------------------------------------------------------------------------


class TestFallback:
    def test_raising_callback_fails_open(self, monkeypatch):
        """``invoke_hook`` isolates each callback; a broken plugin must not brick
        the terminal. Strictness belongs in the plugin, which can catch its own
        errors and return an explicit block."""
        def _explode(hook_name, **kw):
            raise RuntimeError("plugin exploded")

        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _explode)
        assert resolve_pre_exec(**CONTEXT).blocked is False

    def test_raising_callback_through_real_manager_fails_open(self, monkeypatch):
        """End-to-end through the real dispatcher: a raising callback yields no
        results at all, so the resolver sees the absent case."""
        manager = plugins_mod.PluginManager()

        def _bad(**kwargs):
            raise RuntimeError("boom")

        manager._hooks.setdefault(PRE_EXEC_HOOK, []).append(_bad)
        monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
        monkeypatch.setattr(plugins_mod, "_delivery_manager", lambda: manager)
        assert resolve_pre_exec(**CONTEXT).blocked is False

    def test_has_hook_failure_fails_open(self, monkeypatch):
        def _explode(name):
            raise RuntimeError("discovery blew up")

        monkeypatch.setattr("hermes_cli.plugins.has_hook", _explode)
        assert resolve_pre_exec(**CONTEXT).blocked is False


# ---------------------------------------------------------------------------
# Payload contract — only the existing execution context, nothing new resolved
# ---------------------------------------------------------------------------


class TestPayload:
    def test_callback_receives_the_execution_context(self, monkeypatch):
        captured = _fake_hooks(monkeypatch, [])
        resolve_pre_exec(**CONTEXT)
        assert captured["hook_name"] == PRE_EXEC_HOOK
        for key, value in CONTEXT.items():
            assert captured["kwargs"][key] == value

    def test_payload_carries_no_extra_context_fields(self, monkeypatch):
        captured = _fake_hooks(monkeypatch, [])
        resolve_pre_exec(**CONTEXT)
        extra = set(captured["kwargs"]) - set(CONTEXT)
        # invoke_hook stamps telemetry_schema_version itself; the resolver adds nothing.
        assert extra <= {"telemetry_schema_version"}


# ---------------------------------------------------------------------------
# Terminal fire site
# ---------------------------------------------------------------------------


class TestTerminalFireSite:
    def _pre_exec_block(self, monkeypatch, **overrides):
        from tools import terminal_tool

        monkeypatch.setattr(terminal_tool, "gateway_lifecycle_block",
                            overrides.pop("gateway_lifecycle_block", lambda **kw: None))
        monkeypatch.setattr(terminal_tool, "self_repo_block", lambda **kw: None)
        return terminal_tool._pre_exec_block(
            "echo hello", env=object(), env_type="local", cwd="/tmp/project",
            workdir=None, session_key="session-1",
        )

    def test_absent_hook_leaves_the_chain_unchanged(self, monkeypatch):
        _no_hooks(monkeypatch)
        assert self._pre_exec_block(monkeypatch) is None

    def test_allow_lets_the_command_run(self, monkeypatch):
        _fake_hooks(monkeypatch, [{"action": "allow"}])
        assert self._pre_exec_block(monkeypatch) is None

    def test_block_raises_rejected_with_the_blocked_envelope(self, monkeypatch):
        from tools import terminal_tool

        _fake_hooks(monkeypatch, [{"action": "block", "reason": "guard says no"}])
        with pytest.raises(terminal_tool._Rejected) as excinfo:
            self._pre_exec_block(monkeypatch)
        payload = json.loads(excinfo.value.result_json)
        assert payload["exit_code"] == 1
        assert payload["status"] == "blocked"
        assert "guard says no" in payload["error"]
        assert payload["output"] == ""

    def test_hook_runs_after_the_built_in_guards(self, monkeypatch):
        """A host guard is authoritative: when it blocks, the hook never sees the
        command, so a plugin can neither observe nor allow it."""
        from tools import terminal_tool

        def _boom(hook_name, **kw):  # pragma: no cover - the assert is the point
            raise AssertionError("pre_exec fired before the built-in guards blocked")

        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _boom)
        with pytest.raises(terminal_tool._Rejected):
            self._pre_exec_block(
                monkeypatch,
                gateway_lifecycle_block=lambda **kw: json.dumps(
                    {"output": "", "exit_code": 1, "error": "host guard", "status": "error"}),
            )
