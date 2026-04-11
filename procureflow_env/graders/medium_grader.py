"""Deterministic grader for the medium task."""

from app.models import TaskData
from app.scoring import normalize_submission_score
from app.state import RuntimeState


def grade_medium(task: TaskData, runtime_state: RuntimeState) -> float:
    """Grade the vendor-selection task deterministically."""
    if runtime_state.selected_vendor == task.expected_vendor_id:
        raw_score = 1.0
    elif runtime_state.selected_vendor in task.acceptable_vendor_ids:
        raw_score = 0.5
    else:
        raw_score = 0.0
    return normalize_submission_score(raw_score)