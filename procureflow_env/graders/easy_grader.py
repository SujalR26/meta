from app.scoring import normalize_submission_score
from app.models import TaskData
from app.state import RuntimeState

def grade_easy(task: TaskData, runtime_state: RuntimeState) -> float:
    """Grade the easy policy task with soft scoring."""

    if runtime_state.decision == task.expected_decision:
        raw_score = 0.95
    else:
        # partial reasoning still gets something
        raw_score = 0.2

    return normalize_submission_score(raw_score)