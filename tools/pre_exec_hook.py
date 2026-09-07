"""The generic ``pre_exec`` policy hook: an opt-in veto on command execution.

Hermes already guards command execution with built-in, host-owned checks (the
supervised-gateway lifecycle block, the workdir allowlist, the Windows
self-repo guard). Downstream deployments have their own preconditions that
core cannot know about, and today the only way to express one is to fork the
guard chain. This module is the seam: a plugin registers a ``pre_exec``
callback, receives the execution context that the guard chain already has, and
returns a verdict.

Deliberately narrow:

* **Opt-in only.** No configuration key and no environment fallback. With no
  callback registered the resolver short-circuits on :func:`has_hook` and the
  guard chain behaves exactly as it did before this module existed.
* **Surface-agnostic.** The resolver returns a :class:`PreExecDecision`, never
  a wire payload — the calling surface owns how a refusal is rendered. The one
  fire site today is the terminal tool's pre-execution guard chain
  (``tools.terminal_tool._pre_exec_block``); the contract does not assume it.
* **A veto, not an override.** ``allow`` is advisory: it cannot resurrect a
  command a built-in guard already refused (the hook runs last, so it never
  sees one), and it does not veto another callback's ``block``. A plugin can
  only ever narrow what runs.
* **Fail open.** ``invoke_hook`` isolates each callback, so one that raises
  yields no verdict and the command proceeds — a broken plugin degrades to
  absent rather than bricking every command. A plugin that wants strictness
  owns it: catch your own errors and return an explicit ``block``.

Verdict contract — the callback returns ``None`` or a dict:

``{"action": "block", "reason": "<non-empty str>"}``
    Refuse the command; ``reason`` is shown to the operator and the model. A
    block without a usable reason is ignored, so a malformed directive can
    never produce an unexplained refusal.
``{"action": "allow"}``
    Proceed. Advisory (see above).
anything else
    Non-applicable; ignored.

Every registered callback runs. The first valid ``block`` supplies the reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

PRE_EXEC_HOOK = "pre_exec"

ACTION_ALLOW = "allow"
ACTION_BLOCK = "block"

# A refusal reason is rendered into the tool result the model reads; cap it so a
# verbose plugin cannot flood the transcript.
MAX_REASON_CHARS = 2_000


@dataclass(frozen=True)
class PreExecDecision:
    """The resolved verdict. ``reason`` is set only when ``blocked``."""

    blocked: bool = False
    reason: Optional[str] = None


_PROCEED = PreExecDecision()


def _block_reason(result: Any) -> Optional[str]:
    """The reason from a well-formed block directive, else ``None`` (non-applicable)."""
    if not isinstance(result, dict) or result.get("action") != ACTION_BLOCK:
        return None
    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        logger.warning("pre_exec block directive ignored: 'reason' must be a non-empty string (got %r)", reason)
        return None
    return reason.strip()[:MAX_REASON_CHARS]


def resolve_pre_exec(
    *,
    command: str,
    env_type: str,
    cwd: str,
    workdir: Optional[str] = None,
    session_key: str = "",
) -> PreExecDecision:
    """Ask registered ``pre_exec`` callbacks whether *command* may run.

    The keyword arguments are the execution context the guard chain already
    holds; they are passed through verbatim and nothing further is resolved on
    the plugin's behalf. Returns a blocked decision only on an explicit,
    well-formed block directive — every other outcome, including no plugin at
    all and a plugin that raises, proceeds.
    """
    from hermes_cli.plugins import has_hook, invoke_hook

    try:
        if not has_hook(PRE_EXEC_HOOK):
            return _PROCEED
        results = invoke_hook(
            PRE_EXEC_HOOK, command=command, env_type=env_type, cwd=cwd,
            workdir=workdir, session_key=session_key,
        )
    except Exception as exc:
        # Dispatch itself failed (plugin discovery, a wedged manager). The host
        # guards have already had their say; degrade to absent.
        logger.warning("pre_exec hook dispatch failed (%s); proceeding", exc)
        return _PROCEED

    for result in results or ():
        reason = _block_reason(result)
        if reason:
            logger.info("pre_exec hook blocked a command: %s", reason)
            return PreExecDecision(blocked=True, reason=reason)
    return _PROCEED
