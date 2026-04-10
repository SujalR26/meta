"""Utilities for returning submission-safe task scores."""

from __future__ import annotations

from math import isfinite


_EPSILON = 1e-6


def normalize_submission_score(score: float) -> float:
    """Clamp task scores into the strict open interval `(0, 1)`."""
    if not isfinite(score):
        raise ValueError("Score must be finite.")
    if score <= 0.0:
        return _EPSILON
    if score >= 1.0:
        return 1.0 - _EPSILON
    return score
