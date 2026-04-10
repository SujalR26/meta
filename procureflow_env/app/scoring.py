from __future__ import annotations
from math import isfinite

_EPSILON = 1e-6

def normalize_submission_score(score: float) -> float:
    """Ensure score lies strictly in (0, 1)."""
    if not isfinite(score):
        raise ValueError("Score must be finite.")

    # Soft clamp instead of hard edges
    if score <= 0.0:
        return _EPSILON
    if score >= 1.0:
        return 1.0 - _EPSILON

    # Extra safety against rounding issues
    if score < _EPSILON:
        return _EPSILON
    if score > 1.0 - _EPSILON:
        return 1.0 - _EPSILON

    return score