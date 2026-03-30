"""Deterministic grader for the easy task."""

from app.models import TaskData
from app.state import RuntimeState


def grade_easy(task: TaskData, runtime_state: RuntimeState) -> float:
    """Grade the easy policy task deterministically."""
    return 1.0 if runtime_state.decision == task.expected_decision else 0.0
