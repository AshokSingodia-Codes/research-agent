import re

from app.models.schemas import Contradiction, Finding

# Antonym / opposing indicator patterns
OPPOSING_PATTERNS = [
    (r"\b(higher|increases|improved|superior)\b", r"\b(lower|decreases|degraded|inferior)\b"),
    (r"\b(strictly consistent|linearizable)\b", r"\b(eventually consistent|weak consistency)\b"),
    (r"\b(safe|prevents split-brain)\b", r"\b(unsafe|permits split-brain|vulnerable)\b"),
    (r"\b(sub-millisecond|ultra-low latency)\b", r"\b(high latency|multi-second|slow)\b"),
    (r"\b(identical|equal throughput)\b", r"\b(different|divergent throughput|higher throughput)\b"),
]


def detect_contradictions(findings: list[Finding]) -> list[Contradiction]:
    """
    Compares findings across different sources to identify conflicting claims,
    emitting Contradiction records.
    """
    contradictions: list[Contradiction] = []
    seen_pairs = set()

    for i in range(len(findings)):
        for j in range(i + 1, len(findings)):
            f1 = findings[i]
            f2 = findings[j]

            # Compare only across different source URLs or domains
            if f1.source_url == f2.source_url:
                continue

            pair_key = tuple(sorted([f1.id, f2.id]))
            if pair_key in seen_pairs:
                continue

            claim1_lower = f1.claim.lower()
            claim2_lower = f2.claim.lower()

            for pattern_a, pattern_b in OPPOSING_PATTERNS:
                has_a1 = bool(re.search(pattern_a, claim1_lower))
                has_b1 = bool(re.search(pattern_b, claim1_lower))
                has_a2 = bool(re.search(pattern_a, claim2_lower))
                has_b2 = bool(re.search(pattern_b, claim2_lower))

                # If claim1 has pattern A and claim2 has pattern B (or vice versa)
                if (has_a1 and has_b2) or (has_b1 and has_a2):
                    seen_pairs.add(pair_key)
                    contradictions.append(
                        Contradiction(
                            claim_a=f1.claim,
                            claim_b=f2.claim,
                            sources=[f1.source_url, f2.source_url],
                            severity="high" if "split-brain" in claim1_lower or "unsafe" in claim1_lower else "medium",
                            explanation=(
                                f"Discrepancy identified between {f1.domain} and {f2.domain} "
                                f"regarding system properties."
                            ),
                        )
                    )
                    break

    return contradictions
