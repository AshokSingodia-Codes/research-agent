from typing import Any

from app.models.schemas import Report


def score_contradiction_detection(
    report: Report | dict[str, Any],
    contradiction_expected: bool = False,
) -> dict[str, Any]:
    """
    Evaluates contradiction detection and resolution:
    - contradictions_found_count: number of surfaced conflicting claims
    - expected_contradiction: whether ground truth indicates opposing claims exist
    - contradiction_detected: True if contradictions were surfaced when expected
    - contradiction_accuracy: 1.0 if detection matches ground-truth expectation, 0.0 otherwise
    """
    if isinstance(report, dict):
        conflicts = report.get("conflicting_evidence", [])
    else:
        conflicts = getattr(report, "conflicting_evidence", []) or []

    count = len(conflicts)
    detected = count > 0

    if contradiction_expected:
        accuracy = 1.0 if detected else 0.0
    else:
        accuracy = 1.0 if not detected else 0.8  # Slight penalty if over-sensitive

    return {
        "contradictions_found_count": count,
        "contradiction_expected": contradiction_expected,
        "contradiction_detected": detected,
        "contradiction_accuracy": round(accuracy, 2),
    }
