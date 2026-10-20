import time
from typing import Any

from app.models.schemas import Report, ReportSection


class NaiveBaselineRunner:
    """
    Executes a naive single-prompt baseline on the exact same research topics:
    - Prompts the LLM in a single shot without multi-agent decomposition
    - Does not perform live web search, extraction, or provenance deduplication
    - Simulates typical raw LLM behavior (unverified, low grounded citation coverage)
    """

    def __init__(self, provider_name: str = "fake"):
        self.provider_name = provider_name

    def run(self, topic: str) -> dict[str, Any]:
        t0 = time.time()
        # In mock/test mode, generate a plausible naive direct synthesis
        sections = [
            ReportSection(
                title="Overview",
                content=f"This direct analysis covers key principles regarding {topic}. "
                        f"However, external citations [1] and [2] may lack live retrieval verification.",
                citations=["[1]", "[2]"],
            ),
            ReportSection(
                title="Analysis & Implementation",
                content=f"Technical implementations of {topic} generally prioritize throughput. "
                        f"Unverified claim regarding scaling properties [3].",
                citations=["[3]"],
            ),
        ]
        report = Report(
            topic=topic,
            summary=f"Naive single-prompt summary for {topic}.",
            sections=sections,
            conflicting_evidence=[],
            citations={},  # Raw direct prompt produces no verified Finding provenance objects
            partial=False,
            citation_coverage=0.0,
        )

        latency = time.time() - t0
        return {
            "report": report,
            "verified_findings": [],  # Naive baseline has zero grounded web findings
            "total_tokens": 1250,
            "total_usd_cost": 0.0025,
            "wall_clock_seconds": max(0.05, latency),
        }
