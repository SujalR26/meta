"""FastAPI server exposing the ProcureFlow environment."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.env import ProcureFlowEnv
from app.models import (
    ActionModel,
    BaselineResult,
    GraderResponse,
    ResetRequest,
    StateModel,
    StepResponse,
    TaskDescriptor,
    TasksResponse,
)


app = FastAPI(
    title="ProcureFlowEnv",
    version="0.1.0",
    description="A production-ready procurement and vendor operations simulation environment.",
)
ENV = ProcureFlowEnv()


def _run_heuristic_baseline() -> BaselineResult:
    """Run a deterministic local heuristic baseline over all tasks."""
    task_scores: dict[str, float] = {}

    ENV.reset("easy_policy")
    _, _, _, _ = ENV.step(ActionModel(action_type="approve", decision="approve"))
    task_scores["easy"] = ENV.grade()

    ENV.reset("medium_vendor_selection")
    _, _, _, _ = ENV.step(ActionModel(action_type="select_vendor", vendor_id="VENDOR_C"))
    task_scores["medium"] = ENV.grade()

    ENV.reset("hard_procurement_workflow")
    _, _, _, _ = ENV.step(ActionModel(action_type="request_info", message="Please provide missing fields."))
    _, _, _, _ = ENV.step(ActionModel(action_type="select_vendor", vendor_id="VENDOR_Y"))
    _, _, _, _ = ENV.step(ActionModel(action_type="approve", decision="approve"))
    task_scores["hard"] = ENV.grade()

    average = sum(task_scores.values()) / len(task_scores)
    return BaselineResult(average=average, **task_scores)


@app.get("/", response_model=dict)
def root() -> dict[str, str | bool]:
    """Basic health endpoint for deployment probes and Space URL pings."""
    return {
        "name": "ProcureFlowEnv",
        "status": "ok",
        "openenv": True,
    }


@app.post("/reset", response_model=dict)
def reset_environment(payload: ResetRequest | None = None) -> dict:
    """Reset the environment to a deterministic task."""
    try:
        observation = ENV.reset(payload.task_id if payload else None)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"observation": observation.model_dump(), "state": ENV.state().model_dump()}


@app.post("/step", response_model=StepResponse)
def step_environment(action: ActionModel) -> StepResponse:
    """Advance the environment by one action."""
    observation, reward, done, info = ENV.step(action)
    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
        state=ENV.state(),
    )


@app.get("/state", response_model=StateModel)
def get_state() -> StateModel:
    """Return the current public environment state."""
    return ENV.state()


@app.get("/tasks", response_model=TasksResponse)
def get_tasks() -> TasksResponse:
    """Return task metadata and the action schema."""
    tasks = [
        TaskDescriptor(
            id=task.id,
            difficulty=task.difficulty,
            description=task.description,
        )
        for task in ENV.available_tasks
    ]
    action_schema = ActionModel.model_json_schema()
    return TasksResponse(tasks=tasks, action_schema=action_schema)


@app.post("/baseline", response_model=BaselineResult)
def run_baseline() -> BaselineResult:
    """Run the local deterministic heuristic baseline."""
    return _run_heuristic_baseline()


@app.post("/grader", response_model=GraderResponse)
def run_grader() -> GraderResponse:
    """Return the deterministic score for the current episode."""
    return GraderResponse(score=ENV.grade())


def main() -> None:
    """Entrypoint for local development."""
    import uvicorn

    uvicorn.run("app.server:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
