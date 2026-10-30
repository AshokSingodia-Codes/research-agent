import time
from typing import Any

from app.config import settings
from app.llm.client import llm_client
from app.models.schemas import ResearchFindings, StepTrace, SubQuestion

RESEARCHER_SYSTEM_PROMPT = """Extract factual research findings strictly grounded in the provided source material for the assigned sub-question.

Requirements:
1. Extract 1 or 2 concise, high-quality factual findings ONLY.
2. Every claim MUST be directly substantiated by the snippet or extracted text.
3. Keep the snippet short (under 150 characters) and claim direct (under 25 words).
4. For each finding, provide:
   - claim: concise factual assertion
   - source_url: exact source URL
   - domain: source domain name
   - snippet: brief direct quote (max 150 characters)
   - retrieval_timestamp: current timestamp (e.g. 1700000000)
   - content_hash: 16-character alphanumeric string
   - confidence: float between 0.0 and 1.0
5. For the 'status' property, always output "ok" (or "no_results").
"""


def research_subquestion(
    sub_question: SubQuestion,
    source_documents: list[dict[str, Any]] | None = None,
    feedback: str | None = None,
    run_id: str = "run_default",
    parent_step_id: str | None = None,
    budget_guard: Any | None = None,
) -> tuple[ResearchFindings, StepTrace]:
    """
    Researcher specialist: Investigates one SubQuestion using extracted evidence documents.
    """
    docs_summary = ""
    if source_documents:
        for idx, doc in enumerate(source_documents, 1):
            docs_summary += (
                f"\n--- Source Document {idx} ---\n"
                f"URL: {doc.get('url', 'unknown')}\n"
                f"Domain: {doc.get('domain', 'unknown')}\n"
                f"Text: {doc.get('text', '')[:1200]}\n"
            )
    else:
        docs_summary = "No pre-fetched source documents provided. Synthesize initial factual findings."

    feedback_text = f"\nSupervisor Feedback to address: {feedback}" if feedback else ""

    user_prompt = (
        f"Sub-Question ID: {sub_question.id}\n"
        f"Sub-Question: {sub_question.text}\n"
        f"Rationale: {sub_question.rationale}\n"
        f"{feedback_text}\n\n"
        f"Available Evidence Documents:\n{docs_summary}\n\n"
        f"Extract structured findings grounded in these documents."
    )

    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    findings_res, trace = llm_client.generate_with_repair(
        agent_name="researcher",
        messages=messages,
        output_schema=ResearchFindings,
        run_id=run_id,
        parent_step_id=parent_step_id,
        budget_guard=budget_guard,
        model=settings.researcher_model,
    )

    # Ensure sub_question_id matches
    findings_res.sub_question_id = sub_question.id
    for f in findings_res.findings:
        f.sub_question_id = sub_question.id
        if not f.retrieval_timestamp:
            f.retrieval_timestamp = time.time()

    return findings_res, trace
