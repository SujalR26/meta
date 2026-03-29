"""Reward helpers for ProcureFlow."""

from __future__ import annotations

from app.models import ActionModel, TaskData
from app.state import RuntimeState


def is_repeated_action(action: ActionModel, runtime_state: RuntimeState) -> bool:
    """Return True when the action repeats the immediately previous action."""
    if not runtime_state.trace.actions:
        return False
    previous = runtime_state.trace.actions[-1]
    return previous.model_dump() == action.model_dump()


def invalid_action_penalty() -> float:
    """Penalty for invalid or schema-valid but task-invalid actions."""
    return -0.1


def repeated_action_penalty() -> float:
    """Penalty for looping or repeated action submissions."""
    return -0.2


def intermediate_progress_reward() -> float:
    """Reward for correct workflow progress."""
    return 0.2


def vendor_reward(task: TaskData, vendor_id: str | None) -> float:
    """Reward for vendor selection quality."""
    if vendor_id == task.expected_vendor_id:
        return 0.3
    if vendor_id in task.acceptable_vendor_ids:
        return 0.15
    return 0.0


def final_decision_reward(task: TaskData, decision: str | None) -> float:
    """Reward for a correct final decision."""
    if decision == task.expected_decision:
        return 0.5
    return 0.0
