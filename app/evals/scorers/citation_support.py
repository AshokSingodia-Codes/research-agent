import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings
from app.llm.client import llm_client
from app.models.schemas import Finding, Report

logger = logging.getLogger(__name__)


class CitationSupportJudgment(BaseModel):
    is_supported: bool = Field(description="True if the source evidence directly or logically supports the claim")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(default="", description="1-sentence explanation of the support determination")


def _resolve_judge_provider() -> tuple[str, str]:
    """
    Selects an evaluation judge model/provider DIFFERENT from the generation provider.
    Generator default is Groq llama-3.3-70b-versatile.
    """
    gen_provider = settings.writer_provider.lower()

    if gen_provider != "gemini" and settings.gemini_api_key:
        return "gemini", settings.gemini_model
    if gen_provider != "anthropic" and settings.anthropic_api_key:
        return "anthropic", settings.anthropic_model
    if gen_provider != "openrouter" and settings.openrouter_api_key:
        return "openrouter", settings.openrouter_model
    if gen_provider != "grok" and settings.grok_api_key:
        return "grok", settings.grok_model

    # If no alternate provider key is configured, select a distinct model family on Groq
    if gen_provider == "groq":
        return "groq", "llama-3.1-8b-instant"

    return "groq", settings.groq_model


def score_citation_support(
    report: Report | dict[str, Any],
    verified_findings: list[Finding] | list[dict[str, Any]],
    mock: bool = True,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Evaluates whether the cited source text actually factually supports each inline claim.
    Uses an independent LLM judge (different provider/model than generator) or NLI token overlap in mock mode.
    """
    if isinstance(report, dict):
        citations_dict = report.get("citations", {})
    else:
        citations_dict = {
            k: (v.model_dump() if hasattr(v, "model_dump") else v)
            for k, v in report.citations.items()
        }

    findings_map = {}
    for f in verified_findings:
        fid = f.get("id") if isinstance(f, dict) else getattr(f, "id", None)
        url = f.get("source_url") if isinstance(f, dict) else getattr(f, "source_url", None)
        ev = f.get("evidence") or f.get("claim") if isinstance(f, dict) else (getattr(f, "evidence", "") or getattr(f, "claim", ""))
        if fid:
            findings_map[fid] = ev
        if url:
            findings_map[url] = ev

    total_evaluated = 0
    supported_count = 0
    unsupported_count = 0
    evaluations = []

    judge_provider, judge_model = _resolve_judge_provider()

    for cite_key, cite_info in citations_dict.items():
        claim = cite_info.get("claim", "") if isinstance(cite_info, dict) else getattr(cite_info, "claim", "")
        fid = cite_info.get("finding_id") if isinstance(cite_info, dict) else getattr(cite_info, "finding_id", None)
        url = cite_info.get("source_url") if isinstance(cite_info, dict) else getattr(cite_info, "source_url", None)

        evidence = findings_map.get(fid) or findings_map.get(url) or ""

        if not claim:
            continue

        total_evaluated += 1

        if not evidence:
            unsupported_count += 1
            evaluations.append({
                "citation_key": cite_key,
                "claim": claim,
                "is_supported": False,
                "reasoning": "No backing source evidence or finding found.",
            })
            continue

        if mock or settings.is_mock_llm:
            # Deterministic NLI lexical overlap judge in mock mode
            claim_words = set(re.findall(r"\w+", claim.lower()))
            ev_words = set(re.findall(r"\w+", evidence.lower()))
            overlap = len(claim_words.intersection(ev_words)) / max(len(claim_words), 1)
            is_supported = overlap >= 0.35 or len(claim_words.intersection(ev_words)) >= 2
            if is_supported:
                supported_count += 1
            else:
                unsupported_count += 1
            evaluations.append({
                "citation_key": cite_key,
                "claim": claim,
                "is_supported": is_supported,
                "reasoning": f"Mock lexical NLI overlap: {overlap:.1%}",
            })
        else:
            # Independent LLM Judge
            prompt = (
                f"You are an impartial fact-checking judge evaluating source attribution accuracy.\n\n"
                f"Claim: \"{claim}\"\n"
                f"Source Evidence: \"{evidence}\"\n\n"
                f"Does the source evidence factually and directly support the claim?"
            )
            messages = [
                {"role": "system", "content": "You are a factual verification judge."},
                {"role": "user", "content": prompt},
            ]
            try:
                judgment, _ = llm_client.generate_with_repair(
                    agent_name="critic",
                    messages=messages,
                    output_schema=CitationSupportJudgment,
                    run_id=run_id or f"judge_{cite_key}",
                    model=judge_model,
                )
                if judgment.is_supported:
                    supported_count += 1
                else:
                    unsupported_count += 1
                evaluations.append({
                    "citation_key": cite_key,
                    "claim": claim,
                    "is_supported": judgment.is_supported,
                    "reasoning": judgment.reasoning,
                    "judge_model": f"{judge_provider}:{judge_model}",
                })
            except Exception as e:
                logger.warning(f"Judge evaluation failed for {cite_key}: {e}")
                unsupported_count += 1
                evaluations.append({
                    "citation_key": cite_key,
                    "claim": claim,
                    "is_supported": False,
                    "reasoning": f"Judge error: {e}",
                })

    support_ratio = (supported_count / total_evaluated) if total_evaluated > 0 else 0.0

    return {
        "citation_support_ratio": round(support_ratio, 4),
        "supported_citations": supported_count,
        "unsupported_citations": unsupported_count,
        "total_citations_evaluated": total_evaluated,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "evaluations": evaluations,
    }

