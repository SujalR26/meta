"""Deterministic easy procurement task."""

from app.models import TaskData


def build_easy_task() -> TaskData:
    """Build the easy policy-compliance task."""
    return TaskData(
        id="easy_policy",
        difficulty="easy",
        description="Check policy compliance for a single budget against the policy limit.",
        request_id="REQ-EASY-1001",
        item="Office Chairs",
        budget=950,
        policy_limit=1000,
        vendors=[],
        missing_fields=[],
        instructions=(
            "Decide whether the purchase should be approved or escalated based on the policy limit."
        ),
        expected_decision="approve",
    )
