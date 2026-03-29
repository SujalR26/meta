---
title: ProcureFlowEnv
emoji: ??
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - procurement
  - fastapi
  - docker
---

# ProcureFlowEnv

ProcureFlowEnv is a production-oriented OpenEnv environment that simulates procurement and vendor operations workflows. An agent acts as a procurement operations assistant and must validate policy compliance, request missing data, evaluate vendor quotes, and make approval or escalation decisions through a deterministic Gym-style loop.

## Real-World Motivation

Real procurement teams routinely process structured purchase requests under budget controls, vendor quality constraints, and approval policies. This environment models that operational workflow so agents can be trained and evaluated on realistic decision-making patterns instead of abstract toy tasks.

## Observation Space

The environment returns a structured observation with:

- `request_id`: purchase request identifier
- `item`: requested good or service
- `budget`: requested spend amount
- `vendors`: structured vendor quotes
- `policy_limit`: approval threshold
- `missing_fields`: unresolved required inputs
- `task_id`: active task identifier
- `instructions`: short task guidance

## Action Space

The agent submits an `ActionModel` with:

- `action_type`: one of `select_vendor`, `approve`, `reject`, `escalate`, `request_info`
- `vendor_id`: optional vendor identifier for selection
- `decision`: optional explicit decision label
- `message`: optional explanation or request message

## Task Descriptions

### Task 1: Easy Policy Compliance

The agent compares `budget` and `policy_limit` and decides whether to approve or escalate. The scenario is deterministic and grades exact decision correctness.

### Task 2: Medium Vendor Selection

The agent receives three vendor quotes with `price`, `delivery_days`, and `rating`. The goal is to select the lowest-priced vendor whose rating is at least `4.0`.

### Task 3: Hard Multi-Step Workflow

The agent must detect missing information, request it, review revealed vendor quotes, choose a compliant vendor, and then make the final approval decision. Correct intermediate actions are rewarded and graded explicitly.

## Reward Design

Rewards provide non-binary progress:

- Correct intermediate action: `+0.2`
- Correct vendor selection: `+0.3`
- Correct final decision: `+0.5`
- Invalid action: `-0.1`
- Repeated looped action: `-0.2`

The hard task combines progress rewards across information gathering, vendor selection, and final decision making.

## API Endpoints

- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /baseline`
- `POST /grader`

All endpoints return JSON.

## Setup

```bash
cd procureflow_env
python -m venv .venv
pip install -r requirements.txt
uvicorn app.server:app --host 0.0.0.0 --port 7860
```

On Windows PowerShell:

```powershell
cd procureflow_env
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.server:app --host 0.0.0.0 --port 7860
```

## Docker Usage

```bash
docker build -t procureflow-env .
docker run -p 7860:7860 procureflow-env
```

## Testing

Run the automated API test suite with:

```bash
pytest
```

Run a quick deployment smoke test locally with:

```bash
curl http://localhost:7860/
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"easy_policy"}'
```

The tests cover:

- task catalog discovery
- easy-task decision flow
- medium-task partial-credit grading
- hard-task multi-step workflow
- invalid-action penalties
- local deterministic baseline scoring

## OpenEnv Manifest

The environment is packaged with `openenv.yaml` and a `server/app.py` entrypoint wrapper for OpenEnv validation and Hugging Face Spaces deployment compatibility.

## Baseline Scores

Deterministic local baseline endpoint:

```json
{
  "easy": 1.0,
  "medium": 1.0,
  "hard": 1.0,
  "average": 1.0
}
```

## Submission Inference

The submission inference script is [inference.py](/c:/meta/procureflow_env/inference.py). It uses the OpenAI Python client and reads the required variables:

- `API_BASE_URL`: OpenAI-compatible model endpoint
- `MODEL_NAME`: model identifier for inference
- `HF_TOKEN`: API token used as the OpenAI client API key

You can provide these through shell environment variables or a `.env` file in the project root.

Example `.env` file:

```env
API_BASE_URL=https://your-openai-compatible-endpoint/v1
MODEL_NAME=your-model-name
HF_TOKEN=your-token
```

Run submission inference with:

```bash
python inference.py
```

## Hugging Face Space Notes

This README includes Hugging Face Space metadata with `sdk: docker`, `app_port: 7860`, and the `openenv` tag so the repo carries the Space configuration in source control. After creating the Space, verify the deployed root URL returns `200` on `GET /` and that `POST /reset` responds successfully.

## OpenEnv Validation

If the `openenv` CLI is available in your environment, run:

```bash
openenv validate
```

The built-in API endpoint `POST /baseline` remains fully local and does not require any external API key.
