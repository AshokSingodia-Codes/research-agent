import time
from typing import Any, TypeVar

from pydantic import BaseModel

from app.llm.base import (
    InvalidOutputError,
    LLMProvider,
    LLMResponse,
)
from app.llm.pricing import calculate_cost
from app.models.enums import ReviewAction, ToolStatus
from app.models.schemas import (
    CitationKey,
    CritiqueResult,
    Finding,
    Report,
    ReportSection,
    ResearchFindings,
    ResearchPlan,
    ReviewDecision,
    SubQuestion,
)

T = TypeVar("T", bound=BaseModel)


class FakeProvider(LLMProvider):
    """
    Deterministic fake provider for test suites.
    Supports scripted responses, error injection, validation failures to test repair,
    and simulated latency.
    """

    def __init__(self):
        self.call_count: int = 0
        self.artificial_delay_s: float = 0.0
        self.error_to_raise: Exception | None = None
        self.custom_responses: list[Any] = []
        self.canned_plans: dict[str, ResearchPlan] = {}
        self.return_raw_invalid_json: bool = False

    @property
    def provider_name(self) -> str:
        return "fake"

    def set_error(self, err: Exception | None) -> None:
        self.error_to_raise = err

    def queue_response(self, response: Any) -> None:
        self.custom_responses.append(response)

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        output_schema: type[T],
        model: str = "fake-model",
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LLMResponse[T]:
        self.call_count += 1
        t0 = time.time()

        if self.artificial_delay_s > 0:
            time.sleep(self.artificial_delay_s)

        if self.error_to_raise is not None:
            err = self.error_to_raise
            self.error_to_raise = None
            raise err

        if self.return_raw_invalid_json:
            self.return_raw_invalid_json = False
            raise InvalidOutputError("Malformed json response from model", raw_output="{'broken_json': ")

        # Search for first matching response in queue
        matched_idx = None
        for i, item in enumerate(self.custom_responses):
            if isinstance(item, Exception):
                matched_idx = i
                break
            if isinstance(item, output_schema):
                matched_idx = i
                break
            if isinstance(item, dict):
                try:
                    output_schema.model_validate(item)
                    matched_idx = i
                    break
                except Exception:
                    pass

        if matched_idx is not None:
            item = self.custom_responses.pop(matched_idx)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, output_schema):
                parsed = item
            else:
                parsed = output_schema.model_validate(item)
        else:
            parsed = self._generate_default_canned(messages, output_schema)

        latency_ms = (time.time() - t0) * 1000.0
        tokens_in = 350
        tokens_out = 450
        total_tokens = tokens_in + tokens_out
        cost = calculate_cost(self.provider_name, model, tokens_in, tokens_out)

        return LLMResponse(
            parsed=parsed,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw_text=parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else str(parsed),
            provider=self.provider_name,
            model=model,
        )

    def _generate_default_canned(self, messages: list[dict[str, str]], output_schema: type[T]) -> Any:
        last_msg = messages[-1]["content"] if messages else ""

        if output_schema == ResearchPlan:
            # Deterministically create 3-5 distinct sub-questions
            return ResearchPlan(
                topic=last_msg[:120],
                sub_questions=[
                    SubQuestion(
                        id="sq_1",
                        text="What are the core technical architectures and algorithmic mechanisms?",
                        rationale="Establishes fundamental operational principles and component designs.",
                        priority=1,
                        expected_source_types=["documentation", "papers"],
                        depends_on=[],
                    ),
                    SubQuestion(
                        id="sq_2",
                        text="How do real-world production performance benchmarks compare across systems?",
                        rationale="Quantifies empirical throughput, latency, and resource scaling trade-offs.",
                        priority=2,
                        expected_source_types=["benchmarks", "technical_reports"],
                        depends_on=["sq_1"],
                    ),
                    SubQuestion(
                        id="sq_3",
                        text="What failure modes and recovery protocols are observed under network partitions?",
                        rationale="Evaluates distributed reliability, partition tolerance, and crash resilience.",
                        priority=3,
                        expected_source_types=["incident_reports", "specifications"],
                        depends_on=["sq_1"],
                    ),
                    SubQuestion(
                        id="sq_4",
                        text="What trade-offs exist between memory footprint and query latencies in deployment?",
                        rationale="Examines hardware sizing requirements and economic operational costs.",
                        priority=4,
                        expected_source_types=["case_studies", "benchmarks"],
                        depends_on=["sq_2"],
                    ),
                ],
            )

        elif output_schema == ResearchFindings:
            return ResearchFindings(
                sub_question_id="sq_1",
                findings=[
                    Finding(
                        sub_question_id="sq_1",
                        claim="Distributed consensus ensures strong data consistency across clustered nodes.",
                        source_url="https://raft.github.io/raft.pdf",
                        domain="raft.github.io",
                        snippet="Raft decomposes consensus into leader election, log replication, and safety guarantees.",
                        retrieval_timestamp=time.time(),
                        content_hash="hash_raft_spec_001",
                        confidence=0.95,
                        source_quality_score=0.9,
                    ),
                    Finding(
                        sub_question_id="sq_1",
                        claim="Log compaction with periodic snapshots bounds disk growth and enables quick recovery.",
                        source_url="https://etcd.io/docs/v3.5/learning/raft/",
                        domain="etcd.io",
                        snippet="etcd checkpoints state machine state to disk to truncate superseded write-ahead logs.",
                        retrieval_timestamp=time.time(),
                        content_hash="hash_etcd_spec_002",
                        confidence=0.92,
                        source_quality_score=0.85,
                    ),
                ],
                search_queries_used=["consensus architecture and recovery protocols"],
                status=ToolStatus.OK,
            )

        elif output_schema == ReviewDecision:
            return ReviewDecision(
                sub_question_id="sq_1",
                action=ReviewAction.ACCEPT,
                reason="Sub-question answered with robust authoritative primary sources.",
            )

        elif output_schema == Report:
            c1 = CitationKey(
                key="[1]",
                finding_id="fnd_1",
                source_url="https://raft.github.io/raft.pdf",
                domain="raft.github.io",
                claim="Distributed consensus ensures strong data consistency across clustered nodes.",
            )
            c2 = CitationKey(
                key="[2]",
                finding_id="fnd_2",
                source_url="https://etcd.io/docs/v3.5/learning/raft/",
                domain="etcd.io",
                claim="Log compaction bounds disk growth and enables quick recovery.",
            )
            return Report(
                topic=last_msg[:100],
                summary="The evaluated architecture demonstrates high throughput and deterministic crash-recovery protocols [1].",
                sections=[
                    ReportSection(
                        title="Architectural Foundations",
                        content="Clustered topologies rely on leader leases and quorum-based replication to prevent split-brain conditions [1].",
                        citations=["[1]"],
                    ),
                    ReportSection(
                        title="Operational Resilience",
                        content="Storage subsystems implement write-ahead log pruning via incremental snapshots [2].",
                        citations=["[2]"],
                    ),
                ],
                conflicting_evidence=[],
                citations={"[1]": c1, "[2]": c2},
                partial=False,
                citation_coverage=1.0,
            )

        elif output_schema == CritiqueResult:
            return CritiqueResult(
                unsupported_claims=[],
                missing_plan_coverage=[],
                passed=True,
                critique_notes="Report demonstrates sound claim coverage with verified citation keys.",
            )

        # Fallback default
        try:
            return output_schema.model_validate({})
        except Exception:
            raise InvalidOutputError(f"Cannot generate canned instance of {output_schema}")
