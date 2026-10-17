import time
from typing import Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.models.schemas import ResearchPlan, SubQuestion, StepTrace
from src.agents.llm import get_anthropic_chat_model

PLANNER_SYSTEM_PROMPT = """You are a specialized Research Planning agent.
Your objective is to decompose a complex research topic into focused, non-overlapping, and empirically answerable sub-questions.

Requirements:
1. Generate between 2 and {max_subquestions} sub-questions. Do NOT exceed {max_subquestions}.
2. Each sub-question must be concrete, specific, and actionable via web search and technical literature.
3. For each sub-question, specify:
   - id: sequential id, e.g. "sq_1", "sq_2"
   - question: direct research question
   - rationale: why this information is necessary for the overall synthesis
   - expected_evidence: the specific data, benchmarks, architectures, or findings expected to be found
4. Avoid vague queries like "What is X?". Ask targeted questions like "What are the performance trade-offs of X compared to Y in production deployments?".
"""


def plan_research(
    topic: str,
    max_subquestions: int = 4
) -> Tuple[ResearchPlan, StepTrace]:
    """
    Executes the Planner specialist to create a structured ResearchPlan.
    Returns the plan and the StepTrace audit record.
    """
    start_time = time.time()
    clamped_max = min(max_subquestions, settings.max_subquestions)
    model = get_anthropic_chat_model(temperature=0.1)

    plan: ResearchPlan
    tokens_used = 0

    if model is not None:
        try:
            structured_llm = model.with_structured_output(ResearchPlan)
            prompt = PLANNER_SYSTEM_PROMPT.format(max_subquestions=clamped_max)
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Decompose this research topic: {topic}")
            ]
            response = structured_llm.invoke(messages)
            if isinstance(response, ResearchPlan):
                plan = response
                # Ensure sub-questions don't exceed budget
                if len(plan.sub_questions) > clamped_max:
                    plan.sub_questions = plan.sub_questions[:clamped_max]
                tokens_used = 1200  # approximate token usage for planning call
            else:
                plan = _fallback_mock_plan(topic, clamped_max)
                tokens_used = 50
        except Exception as e:
            # Fallback if API fails or quota exceeded
            plan = _fallback_mock_plan(topic, clamped_max)
            tokens_used = 50
    else:
        plan = _fallback_mock_plan(topic, clamped_max)
        tokens_used = 50

    latency_ms = (time.time() - start_time) * 1000.0

    trace = StepTrace(
        step_id=f"plan_{int(start_time)}",
        node_name="planner",
        latency_ms=round(latency_ms, 2),
        tokens_used=tokens_used,
        input_summary=f"Topic: {topic[:100]}",
        output_summary=f"Generated {len(plan.sub_questions)} sub-questions: {[sq.id for sq in plan.sub_questions]}",
        tool_calls=[]
    )

    return plan, trace


def _fallback_mock_plan(topic: str, max_subquestions: int) -> ResearchPlan:
    """Deterministic, high-quality mock research plan for testing and offline runs."""
    sub_questions = [
        SubQuestion(
            id="sq_1",
            question=f"What are the core technical architectures and operational principles of {topic}?",
            rationale="Establishes the foundational mechanics and component design.",
            expected_evidence="Architectural diagrams, state machines, protocol specifications, or core algorithms.",
            status="pending",
            attempts=0
        ),
        SubQuestion(
            id="sq_2",
            question=f"What are the known performance trade-offs, scalability limits, or failure modes of {topic}?",
            rationale="Identifies real-world production limitations and bottlenecks.",
            expected_evidence="Empirical benchmark data, latency figures, memory constraints, and failure recovery metrics.",
            status="pending",
            attempts=0
        ),
        SubQuestion(
            id="sq_3",
            question=f"What are current industry best practices and reference implementations for {topic} in production?",
            rationale="Provides pragmatic deployment patterns and production battle-tested configurations.",
            expected_evidence="Case studies, engineering blog posts, enterprise adoption reports, or open-source reference repos.",
            status="pending",
            attempts=0
        ),
    ]

    return ResearchPlan(
        research_topic=topic,
        sub_questions=sub_questions[:max_subquestions],
        strategy_overview=f"Decomposed '{topic}' into architectural foundations, performance boundaries, and production practices."
    )
