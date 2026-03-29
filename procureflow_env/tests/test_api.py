"""End-to-end API tests for ProcureFlowEnv."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.server import app


client = TestClient(app)


def test_tasks_endpoint_returns_catalog() -> None:
    """The task catalog should expose all deterministic tasks."""
    response = client.get("/tasks")
    assert response.status_code == 200

    payload = response.json()
    task_ids = {task["id"] for task in payload["tasks"]}
    assert task_ids == {
        "easy_policy",
        "medium_vendor_selection",
        "hard_procurement_workflow",
    }
    assert "properties" in payload["action_schema"]


def test_easy_task_flow_scores_perfectly() -> None:
    """The easy task should approve successfully and score 1.0."""
    reset_response = client.post("/reset", json={"task_id": "easy_policy"})
    assert reset_response.status_code == 200

    step_response = client.post(
        "/step",
        json={"action_type": "approve", "decision": "approve"},
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert step_payload["done"] is True
    assert step_payload["reward"] == 0.5

    grader_response = client.post("/grader")
    assert grader_response.status_code == 200
    assert grader_response.json()["score"] == 1.0


def test_medium_task_second_best_vendor_scores_half() -> None:
    """The medium task should award partial credit for the acceptable fallback vendor."""
    reset_response = client.post("/reset", json={"task_id": "medium_vendor_selection"})
    assert reset_response.status_code == 200

    step_response = client.post(
        "/step",
        json={"action_type": "select_vendor", "vendor_id": "VENDOR_A"},
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert step_payload["done"] is True
    assert step_payload["reward"] == 0.15

    grader_response = client.post("/grader")
    assert grader_response.status_code == 200
    assert grader_response.json()["score"] == 0.5


def test_hard_task_multistep_flow_scores_perfectly() -> None:
    """The hard task should reward correct multi-step progress and end at 1.0."""
    reset_response = client.post("/reset", json={"task_id": "hard_procurement_workflow"})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["observation"]["missing_fields"] == ["delivery_address", "vendor_quotes"]

    request_info_response = client.post(
        "/step",
        json={"action_type": "request_info", "message": "Please provide missing fields."},
    )
    assert request_info_response.status_code == 200
    request_info_payload = request_info_response.json()
    assert request_info_payload["reward"] == 0.2
    assert request_info_payload["done"] is False
    assert request_info_payload["observation"]["missing_fields"] == []
    assert len(request_info_payload["observation"]["vendors"]) == 3

    select_vendor_response = client.post(
        "/step",
        json={"action_type": "select_vendor", "vendor_id": "VENDOR_Y"},
    )
    assert select_vendor_response.status_code == 200
    select_vendor_payload = select_vendor_response.json()
    assert select_vendor_payload["reward"] == 0.5
    assert select_vendor_payload["done"] is False

    final_decision_response = client.post(
        "/step",
        json={"action_type": "approve", "decision": "approve"},
    )
    assert final_decision_response.status_code == 200
    final_decision_payload = final_decision_response.json()
    assert final_decision_payload["reward"] == 0.7
    assert final_decision_payload["done"] is True

    grader_response = client.post("/grader")
    assert grader_response.status_code == 200
    assert grader_response.json()["score"] == 1.0


def test_invalid_hard_task_action_is_rejected_with_penalty() -> None:
    """The hard task should penalize invalid sequencing before missing info is resolved."""
    reset_response = client.post("/reset", json={"task_id": "hard_procurement_workflow"})
    assert reset_response.status_code == 200

    step_response = client.post(
        "/step",
        json={"action_type": "approve", "decision": "approve"},
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert step_payload["reward"] == -0.1
    assert step_payload["done"] is False
    assert "error" in step_payload["info"]


def test_local_baseline_endpoint_returns_perfect_scores() -> None:
    """The local heuristic baseline endpoint should return deterministic perfect scores."""
    response = client.post("/baseline")
    assert response.status_code == 200
    assert response.json() == {
        "easy": 1.0,
        "medium": 1.0,
        "hard": 1.0,
        "average": 1.0,
    }
