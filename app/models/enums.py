from enum import StrEnum


class ToolStatus(StrEnum):
    OK = "ok"
    NO_RESULTS = "no_results"
    RATE_LIMITED = "rate_limited"
    PAYWALLED = "paywalled"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    PARSE_ERROR = "parse_error"

    @property
    def status(self) -> str:
        return "success" if self == ToolStatus.OK else str(self.value)


class ReviewAction(StrEnum):
    ACCEPT = "accept"
    RETRY_ONCE = "retry_once"
    SKIP = "skip"
    ABORT = "abort"


class RunStatus(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    RESEARCHING = "researching"
    REVIEWING = "reviewing"
    AWAITING_APPROVAL = "awaiting_approval"
    WRITING = "writing"
    CRITIQUING = "critiquing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_REVIEW = "needs_review"
