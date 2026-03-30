from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from openai import OpenAI

from app.models import ActionModel
from app.server import app


# --- Load env ---
load_dotenv(Path(__file__).resolve().parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")


def _require_env(name: str, value: str | None) -> str:
    if value:
        return value
    raise RuntimeError(f"{name} is required")


# --- Extract JSON safely ---
def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found: {text}")

    return json.loads(match.group())


# --- Prompt ---
def _build_prompt(observation: dict[str, Any], state: dict[str, Any]) -> str:
    return (
        "Return ONLY a JSON object.\n\n"
        f"Observation:\n{json.dumps(observation)}\n\n"
        f"State:\n{json.dumps(state)}\n\n"
        "Rules:\n"
        "- If missing_fields not empty → request_info\n"
        "- Select lowest price vendor with rating >= 4.0\n"
        "- If budget <= policy_limit → approve else escalate\n"
        "- action_type must be one of: select_vendor, approve, reject, escalate, request_info\n"
        "- No explanation, only JSON\n"
    )


# --- Normalize (CORE FIX) ---
def _normalize_action(payload: dict[str, Any]) -> dict[str, Any]:
    action_type = payload.get("action_type") or payload.get("action") or payload.get("type")

    # map invalid names
    mapping = {
        "choose_vendor": "select_vendor",
        "vendor_selection": "select_vendor",
        "evaluate_vendors": "select_vendor",
    }
    action_type = mapping.get(action_type, action_type)

    valid = {"select_vendor", "approve", "reject", "escalate", "request_info"}
    if action_type not in valid:
        action_type = "request_info"

    # --- vendor extraction ---
    vendor_id = payload.get("vendor_id")

    # from selected_vendor
    if not vendor_id and "selected_vendor" in payload:
        selected = payload.get("selected_vendor")
        if isinstance(selected, dict):
            vendor_id = selected.get("vendor_id")

    # from vendors list (deterministic logic)
    if not vendor_id and "vendors" in payload:
        vendors = payload.get("vendors", [])
        if isinstance(vendors, list) and vendors:
            valid_vendors = [v for v in vendors if v.get("rating", 0) >= 4.0]
            if valid_vendors:
                best = min(valid_vendors, key=lambda x: x.get("price", float("inf")))
                vendor_id = best.get("vendor_id") or best.get("name")

    # final safety
    if action_type == "select_vendor" and not vendor_id:
        action_type = "request_info"

    # decision
    decision = payload.get("decision")
    if action_type in {"approve", "reject", "escalate"}:
        decision = action_type

    if action_type == "select_vendor":
        decision = None

    normalized = {
        "action_type": action_type,
        "vendor_id": vendor_id,
        "decision": decision,
        "message": payload.get("message"),
    }

    try:
        return ActionModel(**normalized).model_dump()
    except Exception as e:
        print("NORMALIZATION ERROR:", normalized, e)
        return {
            "action_type": "request_info",
            "vendor_id": None,
            "decision": None,
            "message": "fallback",
        }


# --- LLM call ---
def _next_action(observation: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    model = _require_env("MODEL_NAME", MODEL_NAME)
    token = _require_env("HF_TOKEN", HF_TOKEN)
    base = _require_env("API_BASE_URL", API_BASE_URL)

    client = OpenAI(api_key=token, base_url=base)

    prompt = _build_prompt(observation, state)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )

    text = response.choices[0].message.content
    print("RAW OUTPUT:", text)

    parsed = _extract_json(text)
    return _normalize_action(parsed)


# --- Run task ---
def _run_task(api_client: TestClient, task_id: str) -> float:
    reset = api_client.post("/reset", json={"task_id": task_id})
    reset.raise_for_status()

    payload = reset.json()
    observation = payload["observation"]
    state = payload["state"]

    for _ in range(6):
        action = _next_action(observation, state)

        step = api_client.post("/step", json=action)
        if step.status_code >= 400:
            raise RuntimeError(f"Invalid action: {action} -> {step.text}")

        step_data = step.json()
        observation = step_data["observation"]
        state = step_data["state"]

        if step_data["done"]:
            break

    grader = api_client.post("/grader")
    grader.raise_for_status()

    return float(grader.json()["score"])


# --- Main ---
def run_inference() -> dict[str, float]:
    with TestClient(app) as api_client:
        scores = {
            "easy": _run_task(api_client, "easy_policy"),
            "medium": _run_task(api_client, "medium_vendor_selection"),
            "hard": _run_task(api_client, "hard_procurement_workflow"),
        }

    scores["average"] = sum(scores.values()) / len(scores)
    return scores


if __name__ == "__main__":
    print(json.dumps(run_inference(), indent=2))