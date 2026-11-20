from app.agents.writer import _clean_and_resolve_section
from app.models.schemas import Finding


def test_writer_alias_resolution_and_hallucination_stripping():
    f1 = Finding(
        id="fnd_1",
        sub_question_id="sq_1",
        claim="Raft leader election takes under 150ms.",
        source_url="https://raft.github.io/raft.pdf",
        domain="raft.github.io",
        snippet="Leader election heartbeat timeout is 150-300ms.",
        content_hash="hash1",
    )
    alias_map = {"C1": f1, "1": f1}
    cited_findings = {}

    content = (
        "Raft ensures safe leader election under 150ms [C1]. "
        "Invented claim with phantom citation [C99]. "
        "Mixed claim with valid and invalid [C1][C42]."
    )

    cleaned_text, used_keys = _clean_and_resolve_section(content, alias_map, cited_findings)

    # Valid citation is preserved and mapped to canonical [1]
    assert "[1]" in cleaned_text
    # Phantom citation sentence should be dropped entirely
    assert "phantom" not in cleaned_text
    # Mixed sentence should have the phantom stripped and valid kept
    assert "[C42]" not in cleaned_text
    assert "[C99]" not in cleaned_text
    assert len(cited_findings) == 1
    assert cited_findings["[1]"].id == "fnd_1"

