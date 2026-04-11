from app.scoring import normalize_submission_score
from app.models import TaskData
from app.state import RuntimeState

def grade_easy(task: TaskData, runtime_state: RuntimeState) -> float:
    """Grade the easy policy task deterministically."""

    if runtime_state.decision == task.expected_decision:
        raw_score = 1.0
    else:
        raw_score = 0.0

    return normalize_submission_score(raw_score)