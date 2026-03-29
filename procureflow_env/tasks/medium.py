"""Deterministic medium procurement task."""

from app.models import TaskData, VendorQuote


def build_medium_task() -> TaskData:
    """Build the medium vendor-selection task."""
    return TaskData(
        id="medium_vendor_selection",
        difficulty="medium",
        description="Select the best vendor using price and rating constraints.",
        request_id="REQ-MED-2002",
        item="Laptops",
        budget=4200,
        policy_limit=5000,
        vendors=[
            VendorQuote(vendor_id="VENDOR_A", price=3900, delivery_days=7, rating=4.5),
            VendorQuote(vendor_id="VENDOR_B", price=3600, delivery_days=10, rating=3.9),
            VendorQuote(vendor_id="VENDOR_C", price=3750, delivery_days=8, rating=4.2),
        ],
        missing_fields=[],
        instructions=(
            "Select the lowest-priced vendor that still has a rating of at least 4.0."
        ),
        expected_vendor_id="VENDOR_C",
        acceptable_vendor_ids=["VENDOR_A"],
        expected_decision="approve",
    )
