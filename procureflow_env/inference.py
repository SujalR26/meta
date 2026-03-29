"""Submission inference runner using the OpenAI client and required environment variables."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from openai import OpenAI

from app.models import ActionModel
from app.server import app


load_dotenv(Path(__file__).resolve().parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["select_vendor", "approve", "reject", "escalate", "request_info"],
        },
        "vendor_id": {"type": ["string", "null"]},
        "decision": {"type": ["string", "null"]},
        "message": {"type": ["string", "null"]},
    },
    "required": ["action_type"],
    "additionalProperties": False,
}


def _require_env(name: str, value: str | None) -> str:
    """Require a submission environment variable."""
    if value:
        return value
    raise RuntimeError(f"{name} is required for inference.")


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from model output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
        text = text.strip()
    return json.loads(text)


def _build_prompt(observation: dict[str, Any], state: dict[str, Any]) -> str:
    """Construct a deterministic procurement decision prompt."""
    return (
        "You are a procurement operations assistant. Return exactly one JSON object for the next action.\n\n"
        f"Observation:\n{json.dumps(observation, indent=2)}\n\n"
        f"State:\n{json.dumps(state, indent=2)}\n\n"
        "Decision policy:\n"
        "- If missing_fields is non-empty, action_type must be request_info.\n"
        "- When choosing a vendor, select the lowest-priced vendor with rating >= 4.0.\n"
        "- For policy decisions, if budget <= policy_limit choose approve, otherwise choose escalate.\n"
        "- For approve/reject/escalate, set decision equal to action_type.\n"
        "- Use null for unused optional fields.\n"
    )


def _build_client() -> OpenAI:
    """Construct the OpenAI client from submission environment variables."""
    base_url = _require_env("API_BASE_URL", API_BASE_URL)
    api_key = _require_env("HF_TOKEN", HF_TOKEN)
    return OpenAI(base_url=base_url, api_key=api_key)


def _normalize_action(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize model output into the exact action contract expected by the API."""
    action_type = payload.get("action_type") or payload.get("action") or payload.get("type")
    normalized = {
        "action_type": action_type,
        "vendor_id": payload.get("vendor_id"),
        "decision": payload.get("decision"),
        "message": payload.get("message"),
    }
    if normalized["action_type"] in {"approve", "reject", "escalate"} and not normalized["decision"]:
        normalized["decision"] = normalized["action_type"]
    return ActionModel(**normalized).model_dump()


def _next_action(client: OpenAI, observation: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Request the next action from the configured model endpoint."""
    model_name = _require_env("MODEL_NAME", MODEL_NAME)
    prompt = _build_prompt(observation, state)

    response = client.responses.create(
        model=model_name,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "procureflow_action",
                "schema": ACTION_SCHEMA,
                "strict": True,
            }
        },
    )
    return _normalize_action(_extract_json(response.output_text))


def _run_task(api_client: TestClient, llm_client: OpenAI, task_id: str) -> float:
    """Run one task end to end against the local environment API."""
    reset_response = api_client.post("/reset", json={"task_id": task_id})
    reset_response.raise_for_status()
    payload = reset_response.json()
    observation = payload["observation"]
    state = payload["state"]

    for _ in range(6):
        action = _next_action(llm_client, observation, state)
        step_response = api_client.post("/step", json=action)
        if step_response.status_code >= 400:
            raise RuntimeError(
                f"Inference produced invalid action for task '{task_id}': {action}. Response: {step_response.text}"
            )
        step_payload = step_response.json()
        observation = step_payload["observation"]
        state = step_payload["state"]
        if step_payload["done"]:
            break

    grader_response = api_client.post("/grader")
    grader_response.raise_for_status()
    return float(grader_response.json()["score"])


def run_inference() -> dict[str, float]:
    """Evaluate all tasks and return reproducible scores."""
    llm_client = _build_client()
    with TestClient(app) as api_client:
        scores = {
            "easy": _run_task(api_client, llm_client, "easy_policy"),
            "medium": _run_task(api_client, llm_client, "medium_vendor_selection"),
            "hard": _run_task(api_client, llm_client, "hard_procurement_workflow"),
        }

    scores["average"] = sum(scores.values()) / len(scores)
    return scores


if __name__ == "__main__":
    print(json.dumps(run_inference(), indent=2))
