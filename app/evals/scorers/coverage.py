from typing import Any

from app.models.schemas import Report, ResearchPlan


def score_plan_coverage(
    plan: ResearchPlan | dict[str, Any] | None,
    report: Report | dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Evaluates whether the sub-questions generated in the plan are addressed in the final report:
    - plan_coverage_ratio: fraction of sub-questions represented in report sections
    - total_subquestions: count of subquestions in plan
    - covered_subquestions: count of addressed subquestions
    """
    if not plan or not report:
        return {
            "plan_coverage_ratio": 0.0,
            "total_subquestions": 0,
            "covered_subquestions": 0,
            "uncovered_subquestions": [],
        }

    if isinstance(plan, dict):
        sub_questions = plan.get("sub_questions", [])
    else:
        sub_questions = [sq.model_dump() if hasattr(sq, "model_dump") else sq for sq in plan.sub_questions]

    if isinstance(report, dict):
        sections = report.get("sections", [])
    else:
        sections = [s.model_dump() if hasattr(s, "model_dump") else s for s in report.sections]

    if not sub_questions:
        return {
            "plan_coverage_ratio": 1.0,
            "total_subquestions": 0,
            "covered_subquestions": 0,
            "uncovered_subquestions": [],
        }

    # Aggregate all text from report sections
    combined_report_text = " ".join(
        f"{sec.get('title', '')} {sec.get('content', '')}".lower()
        for sec in sections
    )

    covered = 0
    uncovered = []

    for sq in sub_questions:
        sq_id = sq.get("id", "")
        sq_text = sq.get("text", "").lower()
        # Extract meaningful terms (length >= 4) from sub-question text
        keywords = [word.strip(".,;:?!'\"()") for word in sq_text.split() if len(word) >= 4]

        # A sub-question is considered covered if its id, section title, or key concepts appear in report
        if not keywords:
            covered += 1
            continue

        match_count = sum(1 for kw in keywords if kw in combined_report_text)
        if sq_id.lower() in combined_report_text or match_count >= 1:
            covered += 1
        else:
            uncovered.append(sq_id or sq_text[:30])

    coverage_ratio = round(covered / len(sub_questions), 4)

    return {
        "plan_coverage_ratio": coverage_ratio,
        "total_subquestions": len(sub_questions),
        "covered_subquestions": covered,
        "uncovered_subquestions": uncovered,
    }
