"""Core OpenEnv-compatible procurement environment implementation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.models import ActionModel, ObservationModel, StateModel, TaskData, VendorQuote
from app.reward import (
    final_decision_reward,
    intermediate_progress_reward,
    invalid_action_penalty,
    is_repeated_action,
    repeated_action_penalty,
    vendor_reward,
)
from app.state import RuntimeState
from graders.easy_grader import grade_easy
from graders.hard_grader import grade_hard
from graders.medium_grader import grade_medium
from tasks import TASK_BUILDERS


class ProcureFlowEnv:
    """Deterministic procurement operations simulation environment."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskData] = {task_id: builder() for task_id, builder in TASK_BUILDERS.items()}
        self._runtime_state = RuntimeState()
        self._current_task = deepcopy(self._tasks["easy_policy"])

    @property
    def available_tasks(self) -> list[TaskData]:
        """Return all supported deterministic tasks."""
        return [deepcopy(task) for task in self._tasks.values()]

    def reset(self, task_id: str | None = None) -> ObservationModel:
        """Reset the environment to a deterministic task state."""
        selected_task_id = task_id or "easy_policy"
        if selected_task_id not in self._tasks:
            raise ValueError(f"Unknown task_id '{selected_task_id}'.")

        self._current_task = deepcopy(self._tasks[selected_task_id])
        self._runtime_state = RuntimeState(task_id=selected_task_id)
        return self._build_observation()

    def step(self, action: ActionModel) -> tuple[ObservationModel, float, bool, dict[str, Any]]:
        """Apply an action using Gym-style `(observation, reward, done, info)` semantics."""
        if self._runtime_state.done:
            return (
                self._build_observation(),
                invalid_action_penalty(),
                True,
                {"error": "Episode is already done."},
            )

        info: dict[str, Any] = {"task_id": self._current_task.id}
        reward = 0.0

        if is_repeated_action(action, self._runtime_state):
            self._runtime_state.trace.repeated_actions += 1
            reward += repeated_action_penalty()
            info["loop_detected"] = True

        self._runtime_state.record_action(action)
        self._runtime_state.current_step += 1

        handler = {
            "easy": self._handle_easy_step,
            "medium": self._handle_medium_step,
            "hard": self._handle_hard_step,
        }[self._current_task.difficulty]

        step_reward, step_info = handler(action)
        reward += step_reward
        info.update(step_info)

        observation = self._build_observation()
        return observation, reward, self._runtime_state.done, info

    def state(self) -> StateModel:
        """Return the public current state."""
        return StateModel(
            current_step=self._runtime_state.current_step,
            selected_vendor=self._runtime_state.selected_vendor,
            decision=self._runtime_state.decision,
            done=self._runtime_state.done,
        )

    def grade(self) -> float:
        """Run the deterministic task-specific grader against the current episode."""
        if self._current_task.difficulty == "easy":
            return grade_easy(self._current_task, self._runtime_state)
        if self._current_task.difficulty == "medium":
            return grade_medium(self._current_task, self._runtime_state)
        return grade_hard(self._current_task, self._runtime_state)

    def _build_observation(self) -> ObservationModel:
        """Construct the current observation."""
        return ObservationModel(
            request_id=self._current_task.request_id,
            item=self._current_task.item,
            budget=self._current_task.budget,
            vendors=[VendorQuote(**vendor.model_dump()) for vendor in self._current_task.vendors],
            policy_limit=self._current_task.policy_limit,
            missing_fields=list(self._current_task.missing_fields),
            task_id=self._current_task.id,
            instructions=self._current_task.instructions,
        )

    def _resolve_missing_information(self) -> None:
        """Materialize the hidden information for the hard workflow once requested."""
        updates = self._current_task.resolution_updates
        if "vendors" in updates:
            self._current_task.vendors = [VendorQuote(**vendor) for vendor in updates["vendors"]]
        if "missing_fields" in updates:
            self._current_task.missing_fields = list(updates["missing_fields"])

    def _decision_from_action(self, action: ActionModel) -> str | None:
        """Normalize decision-bearing actions into a decision string."""
        if action.action_type in {"approve", "reject", "escalate"}:
            return action.decision or action.action_type
        return action.decision

    def _handle_easy_step(self, action: ActionModel) -> tuple[float, dict[str, Any]]:
        """Handle one step of the easy policy task."""
        decision = self._decision_from_action(action)
        if action.action_type not in {"approve", "reject", "escalate"} or decision is None:
            self._runtime_state.trace.invalid_actions += 1
            return invalid_action_penalty(), {"error": "A decision action is required for this task."}

        self._runtime_state.decision = decision
        self._runtime_state.done = True
        reward = final_decision_reward(self._current_task, decision)
        return reward, {"expected_decision": self._current_task.expected_decision}

    def _handle_medium_step(self, action: ActionModel) -> tuple[float, dict[str, Any]]:
        """Handle one step of the medium vendor-selection task."""
        if action.action_type != "select_vendor" or not action.vendor_id:
            self._runtime_state.trace.invalid_actions += 1
            return invalid_action_penalty(), {"error": "A vendor selection action is required."}

        candidate_ids = {vendor.vendor_id for vendor in self._current_task.vendors}
        if action.vendor_id not in candidate_ids:
            self._runtime_state.trace.invalid_actions += 1
            return invalid_action_penalty(), {"error": "Unknown vendor id."}

        self._runtime_state.selected_vendor = action.vendor_id
        self._runtime_state.done = True
        reward = vendor_reward(self._current_task, action.vendor_id)
        return reward, {"expected_vendor_id": self._current_task.expected_vendor_id}

    def _handle_hard_step(self, action: ActionModel) -> tuple[float, dict[str, Any]]:
        """Handle one step of the hard multi-stage workflow."""
        progress = self._runtime_state.trace.progress
        info: dict[str, Any] = {}
        reward = 0.0

        if self._current_task.missing_fields:
            if action.action_type != "request_info":
                self._runtime_state.trace.invalid_actions += 1
                return (
                    invalid_action_penalty(),
                    {"error": "Missing information must be requested before continuing."},
                )

            self._resolve_missing_information()
            progress.info_requested = True
            reward += intermediate_progress_reward()
            info["resolved_missing_fields"] = True
            return reward, info

        if self._runtime_state.selected_vendor is None:
            if action.action_type != "select_vendor" or not action.vendor_id:
                self._runtime_state.trace.invalid_actions += 1
                return invalid_action_penalty(), {"error": "Select a vendor after information is complete."}

            candidate_ids = {vendor.vendor_id for vendor in self._current_task.vendors}
            if action.vendor_id not in candidate_ids:
                self._runtime_state.trace.invalid_actions += 1
                return invalid_action_penalty(), {"error": "Unknown vendor id."}

            self._runtime_state.selected_vendor = action.vendor_id
            progress.vendor_selected = True
            reward += intermediate_progress_reward()
            reward += vendor_reward(self._current_task, action.vendor_id)
            info["vendor_selected"] = action.vendor_id
            return reward, info

        decision = self._decision_from_action(action)
        if action.action_type not in {"approve", "reject", "escalate"} or decision is None:
            self._runtime_state.trace.invalid_actions += 1
            return invalid_action_penalty(), {"error": "A final decision action is required."}

        self._runtime_state.decision = decision
        self._runtime_state.done = True
        progress.final_decision_made = True
        reward += intermediate_progress_reward()
        reward += final_decision_reward(self._current_task, decision)
        info["final_decision"] = decision
        return reward, info
