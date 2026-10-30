import uuid
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

from app.models.enums import ReviewAction, RunStatus, ToolStatus


class SubQuestion(BaseModel):
    id: str = Field(description="Unique identifier, e.g., 'sq_1'")
    text: str = Field(description="Concrete sub-question or imperative to research")
    rationale: str = Field(description="Why this information is required for overall synthesis")
    priority: int = Field(default=1, ge=1, le=5, description="Priority ranking from 1 (highest) to 5")
    expected_source_types: list[str] = Field(
        default_factory=lambda: ["academic_papers", "benchmarks"],
        max_length=4,
        description="1 to 3 concise source categories, e.g. ['academic_papers', 'benchmarks']",
    )
    depends_on: list[str] = Field(
        default_factory=list, description="List of other sub-question IDs that must finish before this can execute"
    )


class ResearchPlan(BaseModel):
    topic: str = Field(description="Original research question or topic")
    sub_questions: list[SubQuestion] = Field(
        min_length=1, max_length=8, description="List of non-overlapping, specific research sub-questions"
    )


class Finding(BaseModel):
    id: str = Field(default_factory=lambda: f"fnd_{uuid.uuid4().hex[:8]}")
    sub_question_id: str
    claim: str = Field(description="Specific factual claim or finding")
    source_url: str = Field(description="Canonical URL of the source")
    domain: str = Field(description="Domain name of the source")
    snippet: str = Field(description="Excerpt or snippet supporting the claim")
    extracted_text: str | None = Field(default=None, description="Full extracted readable page content if available")
    retrieval_timestamp: float = Field(description="Unix epoch timestamp when finding was retrieved")
    content_hash: str = Field(description="SHA-256 hash of the extracted content or snippet")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score of the claim")
    source_quality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Source provenance heuristic score")


class ResearchFindings(BaseModel):
    sub_question_id: str
    findings: list[Finding] = Field(default_factory=list)
    search_queries_used: list[str] = Field(default_factory=list)
    status: ToolStatus | str | None = Field(default=ToolStatus.OK)
    error_message: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.status, str):
            try:
                self.status = ToolStatus(self.status)
            except ValueError:
                self.status = ToolStatus.OK
        elif self.status is None:
            self.status = ToolStatus.OK


class ToolResult(BaseModel):
    status: ToolStatus
    findings: list[Finding] = Field(default_factory=list)
    error_message: str | None = None
    query: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __iter__(self):
        return iter((self.findings, self.status))

    def __getitem__(self, item: int):
        return (self.findings, self.status)[item]


class ReviewDecision(BaseModel):
    sub_question_id: str
    action: ReviewAction = Field(description="Supervisory decision: accept, retry_once, skip, or abort")
    reason: str = Field(description="Clear explanation of the review decision")
    follow_up_query: str | None = Field(default=None, description="Refined search query if retry is recommended")


class Contradiction(BaseModel):
    id: str = Field(default_factory=lambda: f"contra_{uuid.uuid4().hex[:8]}")
    claim_a: str = Field(description="First conflicting claim")
    claim_b: str = Field(description="Second conflicting claim")
    sources: list[str] = Field(default_factory=list, description="Source URLs exhibiting conflict")
    severity: Literal["low", "medium", "high"] = "medium"
    explanation: str = Field(description="Why these findings conflict with each other")


class CitationKey(BaseModel):
    key: str = Field(description="Citation key reference, e.g., '[1]'")
    finding_id: str
    source_url: str
    domain: str
    claim: str


class ReportSection(BaseModel):
    title: str
    content: str
    citations: list[str] = Field(
        default_factory=list, description="Inline citation keys used in this section, e.g., ['[1]', '[2]']"
    )


class Report(BaseModel):
    topic: str
    summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    conflicting_evidence: list[Contradiction] = Field(default_factory=list)
    citations: dict[str, CitationKey] = Field(default_factory=dict)
    partial: bool = False
    degradation_reason: str | None = None
    citation_coverage: float = Field(default=1.0, ge=0.0, le=1.0)


class CritiqueResult(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_plan_coverage: list[str] = Field(default_factory=list)
    passed: bool = True
    critique_notes: str = ""


class StepTrace(BaseModel):
    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    run_id: str
    parent_step_id: str | None = None
    agent: str
    provider: str
    model: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    tools_called: list[dict[str, Any]] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0
    usd_cost: float = 0.0
    latency_ms: float = 0.0
    retries: int = 0
    error: str | None = None
    timestamp: float = Field(default_factory=lambda: 0.0)


class ResearchRequest(BaseModel):
    question: str = Field(
        ...,
        validation_alias=AliasChoices("question", "topic"),
        min_length=3,
        max_length=500,
        description="Research topic or question",
    )
    require_approval: bool = Field(default=False, description="Whether to pause for human approval before writing")
    budget_overrides: dict[str, Any] | None = Field(default=None, description="Optional overrides for BudgetGuard")

    @field_validator("question", mode="before")
    @classmethod
    def sanitize_question(cls, v: Any) -> str:
        if isinstance(v, str):
            v = v.strip()
        return v


class ResearchRunResponse(BaseModel):
    run_id: str
    status: RunStatus
    message: str


class ResearchStatusResponse(BaseModel):
    run_id: str
    status: RunStatus
    progress_subquestions: int = 0
    total_subquestions: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    current_node: str | None = None
    report: Report | None = None
    error: str | None = None
    partial: bool = False
    degradation_reason: str | None = None
