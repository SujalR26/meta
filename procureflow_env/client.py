"""Minimal client wrapper for ProcureFlowEnv."""

from __future__ import annotations

from typing import Any

import requests

from models import ActionModel, ObservationModel, StateModel, StepResponse


class ProcureFlowEnvClient:
    """Simple HTTP client for interacting with a ProcureFlow server."""

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_id: str | None = None) -> dict[str, Any]:
        """Reset the remote environment."""
        response = self.session.post(f"{self.base_url}/reset", json={"task_id": task_id})
        response.raise_for_status()
        payload = response.json()
        payload["observation"] = ObservationModel(**payload["observation"])
        payload["state"] = StateModel(**payload["state"])
        return payload

    def step(self, action: ActionModel) -> StepResponse:
        """Step the remote environment."""
        response = self.session.post(f"{self.base_url}/step", json=action.model_dump())
        response.raise_for_status()
        return StepResponse(**response.json())

    def state(self) -> StateModel:
        """Fetch current environment state."""
        response = self.session.get(f"{self.base_url}/state")
        response.raise_for_status()
        return StateModel(**response.json())

    def tasks(self) -> dict[str, Any]:
        """List available tasks."""
        response = self.session.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()

    def grade(self) -> float:
        """Retrieve the current episode score."""
        response = self.session.post(f"{self.base_url}/grader")
        response.raise_for_status()
        return float(response.json()["score"])

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()
