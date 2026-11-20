from app.agents.critic import remove_unsupported_sentences, split_cited_claims
from app.models.schemas import Report, ReportSection


def test_split_cited_claims():
    report = Report(
        topic="Consensus",
        summary="Summary text",
        sections=[
            ReportSection(
                title="Raft",
                content="Raft uses randomized election timers [1]. It achieves strong leader consistency [1][2]. Uncited sentence.",
                citations=["[1]", "[2]"],
            )
        ],
    )
    claims = split_cited_claims(report)
    assert len(claims) == 2
    assert "randomized election timers" in claims[0][0]
    assert claims[0][1] == ["[1]"]
    assert "strong leader consistency" in claims[1][0]
    assert set(claims[1][1]) == {"[1]", "[2]"}


def test_remove_unsupported_sentences():
    report = Report(
        topic="Consensus",
        summary="Summary text",
        sections=[
            ReportSection(
                title="Raft",
                content="Raft uses randomized election timers [1]. FlurbleGlock hypervisor guarantees zero latency [2]. Leader election is robust [1].",
                citations=["[1]", "[2]"],
            )
        ],
    )
    unsupported = ["FlurbleGlock hypervisor guarantees zero latency"]
    cleaned = remove_unsupported_sentences(report, unsupported)
    assert "FlurbleGlock" not in cleaned.sections[0].content
    assert "randomized election timers" in cleaned.sections[0].content
    assert "robust" in cleaned.sections[0].content

