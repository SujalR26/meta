"""State containers for the ProcureFlow environment runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models import ActionModel, EpisodeTrace


@dataclass
class RuntimeState:
    """Mutable runtime state for a single environment episode."""

    task_id: str = "easy_policy"
    current_step: int = 0
    selected_vendor: str | None = None
    decision: str | None = None
    done: bool = False
    trace: EpisodeTrace = field(default_factory=EpisodeTrace)

    def record_action(self, action: ActionModel) -> None:
        """Append an action to the episode trace."""
        self.trace.actions.append(action)
