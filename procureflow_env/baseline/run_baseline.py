"""Compatibility wrapper for the root submission inference script."""

from __future__ import annotations

import json

from inference import run_inference


if __name__ == "__main__":
    print(json.dumps(run_inference(), indent=2))
