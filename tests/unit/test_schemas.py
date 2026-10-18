import pytest
from pydantic import ValidationError

from app.models.schemas import ResearchRequest


def test_research_request_sanitization_and_length_limits():
    """Validates that ResearchRequest question is sanitized (whitespace stripped) and rejects out-of-bound lengths."""
    # 1. Strips leading and trailing whitespace
    req = ResearchRequest(question="   Distributed Consensus Architecture   ")
    assert req.question == "Distributed Consensus Architecture"

    # 2. Rejects questions shorter than 3 characters
    with pytest.raises(ValidationError):
        ResearchRequest(question="ab")

    # 3. Rejects questions longer than 500 characters
    long_q = "A" * 501
    with pytest.raises(ValidationError):
        ResearchRequest(question=long_q)

    # 4. Accepts valid question up to 500 chars
    valid_q = "A" * 500
    req_valid = ResearchRequest(question=valid_q)
    assert len(req_valid.question) == 500


def test_research_request_accepts_question_and_topic_aliases():
    """Validates that ResearchRequest accepts 'question', 'topic', both, and rejects neither."""
    # 1. With question
    r1 = ResearchRequest.model_validate({"question": "Consensus in Raft"})
    assert r1.question == "Consensus in Raft"

    # 2. With topic
    r2 = ResearchRequest.model_validate({"topic": "Consensus in Raft"})
    assert r2.question == "Consensus in Raft"

    # 3. With both
    r3 = ResearchRequest.model_validate({"question": "Question Q", "topic": "Topic T"})
    assert r3.question in ("Question Q", "Topic T")

    # 4. With neither
    with pytest.raises(ValidationError):
        ResearchRequest.model_validate({})
