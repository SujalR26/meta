from app.scoring import normalize_submission_score
from app.models import TaskData
from app.state import RuntimeState

def grade_hard(task: TaskData, runtime_state: RuntimeState) -> float:
    """Grade the multi-step procurement workflow deterministically."""

    score = 0.0
    progress = runtime_state.trace.progress

    if progress.info_requested:
        score += 0.2

    if progress.vendor_selected and runtime_state.selected_vendor == task.expected_vendor_id:
        score += 0.2
    elif progress.vendor_selected and runtime_state.selected_vendor in task.acceptable_vendor_ids:
        score += 0.1

    if progress.final_decision_made and runtime_state.decision == task.expected_decision:
        score += 0.4

    if (
        progress.info_requested
        and progress.vendor_selected
        and progress.final_decision_made
        and runtime_state.selected_vendor == task.expected_vendor_id
        and runtime_state.decision == task.expected_decision
    ):
        score += 0.2

    return normalize_submission_score(score)