from app.scoring import normalize_submission_score
from app.models import TaskData
from app.state import RuntimeState

def grade_hard(task: TaskData, runtime_state: RuntimeState) -> float:
    """Grade multi-step workflow with progressive scoring."""

    score = 0.1  # base score to avoid zero
    progress = runtime_state.trace.progress

    # Step 1: info gathering
    if progress.info_requested:
        score += 0.2

    # Step 2: vendor selection
    if progress.vendor_selected:
        if runtime_state.selected_vendor == task.expected_vendor_id:
            score += 0.25
        elif runtime_state.selected_vendor in task.acceptable_vendor_ids:
            score += 0.15
        else:
            score += 0.05

    # Step 3: final decision
    if progress.final_decision_made:
        if runtime_state.decision == task.expected_decision:
            score += 0.3
        else:
            score += 0.1

    # Bonus for full correct flow (but avoid hitting 1.0)
    if (
        progress.info_requested
        and progress.vendor_selected
        and progress.final_decision_made
        and runtime_state.selected_vendor == task.expected_vendor_id
        and runtime_state.decision == task.expected_decision
    ):
        score += 0.1

    return normalize_submission_score(score)