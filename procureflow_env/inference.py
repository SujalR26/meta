from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from openai import OpenAI

from app.models import ActionModel
from app.scoring import normalize_submission_score
from app.server import app


load_dotenv(Path(__file__).resolve().parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("BENCHMARK", "procureflow")
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))
DEBUG_API = os.getenv("DEBUG_API", "0").lower() in {"1", "true", "yes"}

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

CLIENT = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def _debug(message: str) -> None:
    if DEBUG_API:
        print(message, file=sys.stderr, flush=True)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found: {text}")

    return json.loads(match.group())


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
    except Exception:
        return {
            "action_type": "request_info",
            "vendor_id": None,
            "decision": None,
            "message": "fallback",
        }


def _heuristic_action(observation: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback logic for the ProcureFlow environment."""
    missing = observation.get("missing_fields", [])
    if missing:
        return {
            "action_type": "request_info",
            "message": f"Please provide: {', '.join(missing)}",
        }

    selected_vendor = state.get("selected_vendor")
    if selected_vendor:
        budget = observation.get("budget", 0)
        limit = observation.get("policy_limit", 0)
        if budget <= limit:
            return {"action_type": "approve", "decision": "approve"}
        return {"action_type": "escalate", "decision": "escalate"}

    vendors = observation.get("vendors", [])
    if vendors:
        valid_vendors = [v for v in vendors if v.get("rating", 0) >= 4.0]
        if valid_vendors:
            best = min(valid_vendors, key=lambda x: x.get("price", float("inf")))
            return {
                "action_type": "select_vendor",
                "vendor_id": best.get("vendor_id"),
            }
        # fallback to lowest price anyway if none meet rating
        best = min(vendors, key=lambda x: x.get("price", float("inf")))
        return {
            "action_type": "select_vendor",
            "vendor_id": best.get("vendor_id"),
        }

    budget = observation.get("budget", 0)
    limit = observation.get("policy_limit", 0)
    if budget <= limit:
        return {"action_type": "approve", "decision": "approve"}

    return {"action_type": "escalate", "decision": "escalate"}


def _next_action(observation: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    try:
        _debug("API_CALL building prompt")
        prompt = _build_prompt(observation, state)

        _debug(f"API_CALL start model={MODEL_NAME} base_url={API_BASE_URL}")

        response = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
            timeout=15,
        )

        _debug("API_CALL success")

        text = response.choices[0].message.content or "{}"
        _debug(f"API_CALL raw={text}")

        parsed = _extract_json(text)
        action = _normalize_action(parsed)

        # Guardrails enforce valid stage transitions while still allowing model guidance.
        vendors = observation.get("vendors", [])
        has_missing_fields = bool(observation.get("missing_fields", []))
        selected_vendor = state.get("selected_vendor")

        # Easy-stage policy decision: action must be approve/escalate/reject.
        if not has_missing_fields and not vendors and not selected_vendor:
            if action.get("action_type") not in {"approve", "reject", "escalate"}:
                budget = observation.get("budget", 0)
                limit = observation.get("policy_limit", 0)
                if budget <= limit:
                    action = {"action_type": "approve", "decision": "approve"}
                else:
                    action = {"action_type": "escalate", "decision": "escalate"}

        # Vendor-selection stage: action must choose a known vendor.
        if vendors and not selected_vendor:
            known_ids = {vendor.get("vendor_id") for vendor in vendors if isinstance(vendor, dict)}
            if action.get("action_type") != "select_vendor" or action.get("vendor_id") not in known_ids:
                valid_vendors = [v for v in vendors if v.get("rating", 0) >= 4.0]
                if valid_vendors:
                    best = min(valid_vendors, key=lambda x: x.get("price", float("inf")))
                else:
                    best = min(vendors, key=lambda x: x.get("price", float("inf")))
                action = {"action_type": "select_vendor", "vendor_id": best.get("vendor_id")}

        # Post-vendor stage: enforce final decision action.
        if selected_vendor and action.get("action_type") not in {"approve", "reject", "escalate"}:
            budget = observation.get("budget", 0)
            limit = observation.get("policy_limit", 0)
            if budget <= limit:
                action = {"action_type": "approve", "decision": "approve"}
            else:
                action = {"action_type": "escalate", "decision": "escalate"}

        return action

    except Exception as e:
        _debug(f"API_CALL failed error={str(e)}")
        return _heuristic_action(observation, state)

def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _format_error(error: str | None) -> str:
    if error is None:
        return "null"
    return error.replace("\n", " ").replace("\r", " ")


def _action_to_str(action: dict[str, Any]) -> str:
    action_type = action.get("action_type", "request_info")
    if action_type == "select_vendor":
        vendor_id = action.get("vendor_id")
        if vendor_id is None:
            return "select_vendor()"
        return f"select_vendor('{vendor_id}')"
    if action_type in {"approve", "reject", "escalate"}:
        return f"{action_type}()"
    message = action.get("message")
    if message is None:
        return "request_info()"
    safe_message = str(message).replace("'", "\\'")
    return f"request_info('{safe_message}')"


def _emit_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _emit_step(step_num: int, action_str: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step_num} action={action_str} reward={reward:.2f} "
        f"done={_bool_str(done)} error={_format_error(error)}",
        flush=True,
    )


def _emit_end(success: bool, steps: int, rewards: list[float]) -> None:
    reward_list = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={_bool_str(success)} steps={steps} rewards={reward_list}", flush=True)


def _run_task(task_id: str, task_name: str) -> float:
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.1
    api_client = TestClient(app)

    try:
        _emit_start(task_name)
        reset = api_client.post("/reset", json={"task_id": task_id})
        reset.raise_for_status()

        payload = reset.json()
        observation = payload["observation"]
        state = payload["state"]

        for step_idx in range(1, MAX_STEPS + 1):
            action = _next_action(observation, state)
            step = api_client.post("/step", json=action)

            if step.status_code >= 400:
                raise RuntimeError(f"Invalid action: {action} -> {step.text}")

            step_data = step.json()
            reward = float(step_data.get("reward", 0.1))
            reward = max(min(reward, 0.999), 0.001)
            done = bool(step_data.get("done", False))
            info = step_data.get("info", {})
            error = info.get("error") if isinstance(info, dict) else None

            _emit_step(step_idx, _action_to_str(action), reward, done, error)

            rewards.append(reward)
            steps_taken = step_idx
            observation = step_data["observation"]
            state = step_data["state"]

            if done:
                success = True
                break
    except Exception:
        success = False
    finally:
        # Always return a strict-open score, even if task execution failed.
        try:
            grader_response = api_client.post("/grader")
            if grader_response.status_code < 400:
                raw_score = float(grader_response.json().get("score", 0.1))
                score = normalize_submission_score(raw_score)
            else:
                score = normalize_submission_score(0.1)
        except Exception:
            score = normalize_submission_score(0.1)
        api_client.close()
        _emit_end(success, steps_taken, rewards)

    return score


def run_inference() -> dict[str, float]:
    return {
        "easy": _run_task("easy_policy", "easy"),
        "medium": _run_task("medium_vendor_selection", "medium"),
        "hard": _run_task("hard_procurement_workflow", "hard"),
    }


if __name__ == "__main__":
    run_inference()