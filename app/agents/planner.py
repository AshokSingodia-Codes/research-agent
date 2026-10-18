from typing import Any

from app.config import settings
from app.llm.client import llm_client
from app.models.schemas import ResearchPlan, StepTrace

PLANNER_SYSTEM_PROMPT = """Decompose the user's research question into focused, non-overlapping, and empirically answerable sub-questions.

Requirements:
1. Generate between 3 and 8 sub-questions.
2. Each sub-question must be specific, concrete, at least 6 words long, and end in a question mark or imperative clause.
3. Sub-questions must not duplicate or significantly overlap in semantic scope.
4. Specify:
   - id: sequential identifier ('sq_1', 'sq_2', etc.)
   - text: the concrete sub-question
   - rationale: why this is required to answer the main topic
   - priority: integer 1 (highest) to 5 (lowest)
   - expected_source_types: list of source categories (e.g. academic_papers, documentation, benchmarks)
   - depends_on: list of prerequisite sub-question IDs if this sub-question depends on another
"""


def plan_research(
    topic: str,
    run_id: str = "run_default",
    parent_step_id: str | None = None,
    budget_guard: Any | None = None,
    max_subquestions: int | None = None,
) -> tuple[ResearchPlan, StepTrace]:
    """
    Planner specialist: Decomposes a research question into a structured ResearchPlan.
    """
    limit = max_subquestions or settings.max_subquestions
    user_prompt = (
        f"Research Question: {topic}\n\n"
        f"Generate between 3 and {limit} non-overlapping, concrete sub-questions with dependencies."
    )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    plan, trace = llm_client.generate_with_repair(
        agent_name="planner",
        messages=messages,
        output_schema=ResearchPlan,
        run_id=run_id,
        parent_step_id=parent_step_id,
        budget_guard=budget_guard,
        model=settings.planner_model,
    )

    # Enforce topic and limit clamp
    plan.topic = topic
    if len(plan.sub_questions) > limit:
        plan.sub_questions = plan.sub_questions[:limit]

    return plan, trace
