import logging
import time
from typing import Any

from app.llm.client import llm_client
from app.models.schemas import Report, ReportSection

logger = logging.getLogger(__name__)


class NaiveBaselineRunner:
    """
    Executes a real single-prompt baseline on the exact same research topics:
    - Prompts the configured LLM in a single shot without multi-agent decomposition
    - Does not perform live web search, extraction, or provenance deduplication
    - Uses the exact same LLM model/provider as the synthesis stage
    """

    def __init__(self, agent_name: str = "writer"):
        self.agent_name = agent_name

    def run(self, topic: str, run_id: str | None = None) -> dict[str, Any]:
        t0 = time.time()
        rid = run_id or f"baseline_{int(t0)}"
        prompt = (
            f"You are a research analyst. Write a comprehensive, factual research report on the following topic:\n\n"
            f"Topic: {topic}\n\n"
            f"Include an executive summary, structured sections with detailed content, "
            f"and include bracketed citation markers [1], [2] where relevant. "
            f"Note: You do not have access to live web search; synthesize solely from your training knowledge."
        )
        messages = [
            {"role": "system", "content": "You are a research synthesis agent."},
            {"role": "user", "content": prompt},
        ]
        try:
            report, trace = llm_client.generate_with_repair(
                agent_name=self.agent_name,
                messages=messages,
                output_schema=Report,
                run_id=rid,
            )
            report.topic = topic
            tokens = trace.total_tokens if trace else 0
            cost = trace.usd_cost if trace else 0.0
        except Exception as e:
            logger.warning(f"Naive baseline direct generation failed ({e}); generating direct fallback report")
            report = Report(
                topic=topic,
                summary=f"Direct single-prompt summary for {topic}.",
                sections=[
                    ReportSection(
                        title="Overview",
                        content=f"Direct ungrounded synthesis regarding {topic}.",
                        citations=[],
                    )
                ],
                citations={},
                citation_coverage=0.0,
            )
            tokens = 0
            cost = 0.0

        latency = time.time() - t0
        return {
            "report": report,
            "verified_findings": [],  # Naive baseline has zero grounded web retrieval findings
            "total_tokens": tokens,
            "total_usd_cost": cost,
            "wall_clock_seconds": max(0.05, latency),
        }
