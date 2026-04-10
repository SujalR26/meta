"""Typed API and environment models for ProcureFlow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VendorQuote(BaseModel):
    """Represents a vendor quote submitted for a purchase request."""

    vendor_id: str = Field(..., description="Unique vendor identifier.")
    price: int = Field(..., ge=0, description="Quoted price in whole currency units.")
    delivery_days: int = Field(..., ge=0, description="Estimated delivery time in days.")
    rating: float = Field(..., ge=0.0, le=5.0, description="Vendor quality rating.")


class ObservationModel(BaseModel):
    """Observation returned to the agent after reset and step calls."""

    request_id: str
    item: str
    budget: int = Field(..., ge=0)
    vendors: list[VendorQuote] = Field(default_factory=list)
    policy_limit: int = Field(..., ge=0)
    missing_fields: list[str] = Field(default_factory=list)
    task_id: str
    instructions: str


class ActionModel(BaseModel):
    """Action submitted by the agent."""

    action_type: Literal["select_vendor", "approve", "reject", "escalate", "request_info"]
    vendor_id: str | None = None
    decision: str | None = None
    message: str | None = None


class RewardModel(BaseModel):
    """Reward payload returned by the server."""

    reward: float


class StateModel(BaseModel):
    """Public state exposed by the environment."""

    current_step: int = Field(..., ge=0)
    selected_vendor: str | None = None
    decision: str | None = None
    done: bool = False


class StepResponse(BaseModel):
    """Response returned by the environment step endpoint."""

    observation: ObservationModel
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
    state: StateModel


class ResetRequest(BaseModel):
    """Optional task selector for environment reset."""

    task_id: str | None = None


class BaselineResult(BaseModel):
    """Score bundle returned by the heuristic baseline endpoint."""

    easy: float
    medium: float
    hard: float
    average: float


class TaskDescriptor(BaseModel):
    """Describes a supported task in the environment."""

    id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str


class TasksResponse(BaseModel):
    """Task catalog returned by the environment."""

    tasks: list[TaskDescriptor]
    action_schema: dict[str, Any]


class GraderResponse(BaseModel):
    """Normalized deterministic grader response."""

    score: float = Field(..., gt=0.0, lt=1.0)


class WorkflowProgress(BaseModel):
    """Internal workflow progress for the hard scenario."""

    info_requested: bool = False
    vendor_selected: bool = False
    final_decision_made: bool = False


class TaskData(BaseModel):
    """Internal task configuration used by the environment runtime."""

    id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    request_id: str
    item: str
    budget: int
    policy_limit: int
    vendors: list[VendorQuote] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    instructions: str
    expected_decision: str | None = None
    expected_vendor_id: str | None = None
    acceptable_vendor_ids: list[str] = Field(default_factory=list)
    resolution_updates: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class EpisodeTrace(BaseModel):
    """Internal history snapshot used by graders and debugging."""

    actions: list[ActionModel] = Field(default_factory=list)
    repeated_actions: int = 0
    invalid_actions: int = 0
    progress: WorkflowProgress = Field(default_factory=WorkflowProgress)
