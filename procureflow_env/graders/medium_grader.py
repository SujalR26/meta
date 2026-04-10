from app.scoring import normalize_submission_score
from app.models import TaskData
from app.state import RuntimeState

def grade_medium(task: TaskData, runtime_state: RuntimeState) -> float:
    """Grade vendor selection with graded confidence."""

    if runtime_state.selected_vendor == task.expected_vendor_id:
        raw_score = 0.9
    elif runtime_state.selected_vendor in task.acceptable_vendor_ids:
        raw_score = 0.6
    elif runtime_state.selected_vendor is not None:
        raw_score = 0.3
    else:
        raw_score = 0.1

    return normalize_submission_score(raw_score)