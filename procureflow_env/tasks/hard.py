"""Deterministic hard procurement task."""

from app.models import TaskData


def build_hard_task() -> TaskData:
    """Build the hard multi-step procurement workflow task."""
    return TaskData(
        id="hard_procurement_workflow",
        difficulty="hard",
        description=(
            "Detect missing fields, request information, select a compliant vendor, and make a final decision."
        ),
        request_id="REQ-HARD-3003",
        item="Cloud Backup Subscription",
        budget=2800,
        policy_limit=3000,
        vendors=[],
        missing_fields=["delivery_address", "vendor_quotes"],
        instructions=(
            "First request the missing information, then evaluate vendors, then complete the final decision."
        ),
        expected_vendor_id="VENDOR_Y",
        expected_decision="approve",
        resolution_updates={
            "vendors": [
                {"vendor_id": "VENDOR_X", "price": 2650, "delivery_days": 14, "rating": 4.8},
                {"vendor_id": "VENDOR_Y", "price": 2500, "delivery_days": 10, "rating": 4.4},
                {"vendor_id": "VENDOR_Z", "price": 2350, "delivery_days": 21, "rating": 3.8},
            ],
            "missing_fields": [],
        },
    )
