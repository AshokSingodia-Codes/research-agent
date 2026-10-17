import time
from typing import List, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.config import settings
from src.models.schemas import (
    SubQuestion,
    ResearcherOutput,
    EvidenceRecord,
    SearchStatus,
    StepTrace,
)
from src.tools import search
from src.agents.llm import get_anthropic_chat_model

RESEARCHER_SYNTHESIS_PROMPT = """You are a technical Researcher Specialist.
Given a specific research sub-question, reviewer feedback (if any), and raw search evidence, your task is to extract clear, factual evidence records with strict source attribution.

Rules:
1. Every claim MUST be grounded in the provided search snippet or extracted text.
2. Preserve exact source URLs and domain names.
3. If search results are insufficient or contradictory, explicitly note that.
4. Return a structured list of EvidenceRecord items and a concise synthesis summary.
"""


class _SynthesisResult(BaseModel):
    findings: List[EvidenceRecord] = Field(default_factory=list)
    summary: str = Field(description="Concise synthesis answering the sub-question")


def research_subquestion(
    sub_question: SubQuestion,
    feedback: Optional[str] = None,
    max_searches: int = 1
) -> Tuple[ResearcherOutput, StepTrace]:
    """
    Executes a single sub-question research pass using search tools and extraction.
    Returns: (ResearcherOutput, StepTrace)
    """
    start_time = time.time()
    tool_calls_trace = []

    # 1. Formulate search query
    query = sub_question.question
    if feedback:
        query = f"{sub_question.question} {feedback[:80]}"

    # 2. Execute search and extraction tool
    t0 = time.time()
    raw_findings, search_status = search.search_and_extract(
        query=query,
        max_results=3,
        max_per_domain=2,
        fetch_full_page=True
    )
    tool_latency = (time.time() - t0) * 1000.0

    tool_calls_trace.append({
        "tool": "search_and_extract",
        "query": query,
        "latency_ms": round(tool_latency, 2),
        "status": search_status.status,
        "result_count": len(raw_findings)
    })

    # If search failed or returned 0 results, return structured status immediately
    if search_status.status != "success" or not raw_findings:
        latency_ms = (time.time() - start_time) * 1000.0
        output = ResearcherOutput(
            sub_question_id=sub_question.id,
            findings=[],
            search_status=search_status,
            searches_executed=1,
            summary=f"Search yielded no usable evidence: {search_status.status} ({search_status.error_message or 'empty'})"
        )
        trace = StepTrace(
            step_id=f"research_{sub_question.id}_{int(start_time)}",
            node_name="researcher",
            latency_ms=round(latency_ms, 2),
            tokens_used=10,
            input_summary=f"SubQuestion [{sub_question.id}]: {sub_question.question[:80]}",
            output_summary=f"Search failed with status '{search_status.status}'",
            tool_calls=tool_calls_trace
        )
        return output, trace

    # 3. Process & synthesize findings
    model = get_anthropic_chat_model(temperature=0.1)
    tokens_used = 0
    findings = raw_findings
    summary = f"Identified {len(raw_findings)} primary evidence sources answering: {sub_question.question}"

    if model is not None:
        try:
            evidence_context = "\n\n".join([
                f"[Source: {r.source_url} ({r.domain})]\nSnippet: {r.snippet}\nExtracted: {(r.extracted_text or '')[:600]}"
                for r in raw_findings
            ])
            messages = [
                SystemMessage(content=RESEARCHER_SYNTHESIS_PROMPT),
                HumanMessage(
                    content=f"Sub-question: {sub_question.question}\n"
                            f"Expected Evidence: {sub_question.expected_evidence}\n"
                            f"Reviewer Feedback: {feedback or 'None'}\n\n"
                            f"Available Evidence Sources:\n{evidence_context}"
                )
            ]
            structured_llm = model.with_structured_output(_SynthesisResult)
            synth_resp = structured_llm.invoke(messages)
            if isinstance(synth_resp, _SynthesisResult) and synth_resp.findings:
                findings = synth_resp.findings
                summary = synth_resp.summary
                tokens_used = 1800
        except Exception:
            # Keep raw extracted findings if LLM synthesis fails
            tokens_used = 50
    else:
        tokens_used = 50

    latency_ms = (time.time() - start_time) * 1000.0

    output = ResearcherOutput(
        sub_question_id=sub_question.id,
        findings=findings,
        search_status=search_status,
        searches_executed=1,
        summary=summary
    )

    trace = StepTrace(
        step_id=f"research_{sub_question.id}_{int(start_time)}",
        node_name="researcher",
        latency_ms=round(latency_ms, 2),
        tokens_used=tokens_used,
        input_summary=f"SubQuestion [{sub_question.id}]: {sub_question.question[:80]}",
        output_summary=f"Collected {len(findings)} findings across {[f.domain for f in findings]}",
        tool_calls=tool_calls_trace
    )

    return output, trace
