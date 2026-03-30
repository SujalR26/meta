"""Task registry for the ProcureFlow environment."""

from tasks.easy import build_easy_task
from tasks.hard import build_hard_task
from tasks.medium import build_medium_task

TASK_BUILDERS = {
    "easy_policy": build_easy_task,
    "medium_vendor_selection": build_medium_task,
    "hard_procurement_workflow": build_hard_task,
}
