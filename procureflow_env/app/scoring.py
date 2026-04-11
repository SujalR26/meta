"""Scoring utilities for ProcureFlow grader responses."""

from __future__ import annotations


def normalize_submission_score(score: float) -> float:
    """Ensure score is strictly within open interval (0, 1).
    
    Maps boundary values to values just inside the interval to satisfy
    validation requirements: score > 0.0 and score < 1.0
    """
    epsilon = 1e-6
    
    if score <= 0.0:
        return epsilon
    if score >= 1.0:
        return 1.0 - epsilon
    
    # Extra safety against edge cases from floating point arithmetic
    if score < epsilon:
        return epsilon
    if score > 1.0 - epsilon:
        return 1.0 - epsilon
    
    return score
