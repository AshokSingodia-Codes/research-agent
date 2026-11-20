"""Pure-Python ranked retrieval and graded relevance metrics for evaluation.

Provides offline, deterministic calculation of:
- Precision@k
- Recall@k
- nDCG@k (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- MAP / Average Precision
- Graded Relevance Metrics with human/curated judgments
- Topic Aspect Coverage
- Sub-question lexical coverage
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

TITLE_JACCARD = 0.8
DEFAULT_KS = (5, 10, 20)


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if obj.get(name) is not None:
                return obj[name]
    else:
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def normalize_title(value: Any) -> str:
    """Normalize a paper or article title for robust matching."""
    text = str(value or "")
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def normalize_doi(value: Any) -> str:
    """Normalize DOI to standard lowercase identifier."""
    doi = str(value or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi.removeprefix("doi:").strip().rstrip(".")


def normalize_arxiv(value: Any) -> str:
    """Normalize arXiv identifier to clean ID without version suffix."""
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if raw.startswith("http"):
        parsed = urlparse(raw)
        raw = parsed.path.rsplit("/", 1)[-1]
    raw = re.sub(r"^(arxiv:|abs/|pdf/)", "", raw)
    raw = re.sub(r"\.pdf$", "", raw)
    raw = re.sub(r"v\d+$", "", raw)
    return raw.strip()


def match_paper(must_or_judgment: Mapping[str, Any], paper: Any, title_jaccard: float = TITLE_JACCARD) -> bool:
    """Match a judgment or known-item entry to a retrieved paper or finding using IDs first, then title Jaccard."""
    must_arxiv = normalize_arxiv(must_or_judgment.get("arxiv_id") or must_or_judgment.get("arxiv"))
    paper_arxiv = normalize_arxiv(_get(paper, "arxiv_id", "arxivId", "arxiv"))
    if must_arxiv and paper_arxiv and must_arxiv == paper_arxiv:
        return True

    must_doi = normalize_doi(must_or_judgment.get("doi"))
    paper_doi = normalize_doi(_get(paper, "doi"))
    if must_doi and paper_doi and must_doi == paper_doi:
        return True

    a = set(normalize_title(must_or_judgment.get("title", "")).split())
    b = set(normalize_title(_get(paper, "title", "paper_title", "claim", default="")).split())
    return bool(a and b) and len(a & b) / len(a | b) >= title_jaccard


def precision_at_k(must_cite: Sequence[Mapping[str, Any]], ranked: Sequence[Any], k: int) -> float | None:
    """Precision@k against known-item / must-cite papers."""
    if not must_cite or k <= 0 or not ranked:
        return None
    top = ranked[:k]
    hits = sum(1 for p in top if any(match_paper(m, p) for m in must_cite))
    return round(hits / k, 4)


def recall_at_k(must_cite: Sequence[Mapping[str, Any]], ranked: Sequence[Any], k: int) -> float | None:
    """Recall@k of must-cite items within top-k ranked candidates."""
    if not must_cite or k <= 0:
        return None
    top = ranked[:k]
    hits = sum(1 for m in must_cite if any(match_paper(m, p) for p in top))
    return round(hits / len(must_cite), 4)


def ndcg_at_k(must_cite: Sequence[Mapping[str, Any]], ranked: Sequence[Any], k: int) -> float | None:
    """Normalized Discounted Cumulative Gain at rank k (binary relevance: 1 if must-cite, 0 otherwise)."""
    if not must_cite or k <= 0 or not ranked:
        return None
    top = ranked[:k]
    dcg = sum(
        (1.0 if any(match_paper(m, p) for m in must_cite) else 0.0) / math.log2(i + 2)
        for i, p in enumerate(top)
    )
    ideal_count = min(len(must_cite), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    if idcg <= 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def mean_reciprocal_rank(must_cite: Sequence[Mapping[str, Any]], ranked: Sequence[Any]) -> float | None:
    """Reciprocal rank of first matching must-cite item."""
    if not must_cite or not ranked:
        return None if not must_cite else 0.0
    for rank, p in enumerate(ranked, 1):
        if any(match_paper(m, p) for m in must_cite):
            return round(1.0 / rank, 4)
    return 0.0


# Alias for compatibility
mrr = mean_reciprocal_rank


def average_precision(must_cite: Sequence[Mapping[str, Any]], ranked: Sequence[Any]) -> float | None:
    """Average Precision (AP) across full ranked candidate list."""
    if not must_cite:
        return None
    if not ranked:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, p in enumerate(ranked, 1):
        if any(match_paper(m, p) for m in must_cite):
            hits += 1
            precision_sum += hits / rank
    return round(precision_sum / len(must_cite), 4)


# Alias
mean_average_precision = average_precision


def coverage(must_cite: Sequence[Mapping[str, Any]], candidates: Sequence[Any]) -> tuple[float | None, list[str]]:
    """Fraction of must-cite items retrieved in candidate pool, and list of missed items."""
    if not must_cite:
        return None, []
    missed = [
        str(m.get("title") or m.get("arxiv_id") or m.get("doi") or "unknown")
        for m in must_cite
        if not any(match_paper(m, p) for p in candidates)
    ]
    score = round(1.0 - (len(missed) / len(must_cite)), 4)
    return score, missed


def graded_relevance_metrics(
    judgments: Sequence[Mapping[str, Any]] | None,
    ranked_papers: Sequence[Any],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """Computes graded relevance metrics (Precision, Recall, nDCG, MRR, MAP) from human or curated judgments.

    Grades: 0 = not relevant, 1 = partially/background relevant, 2 = directly relevant.
    """
    judged = list(judgments or [])
    valid = []
    seen = set()
    for j in judged:
        if not isinstance(j, Mapping) or not j.get("title"):
            continue
        arxiv_key = normalize_arxiv(j.get("arxiv_id"))
        doi_key = normalize_doi(j.get("doi"))
        title_key = normalize_title(j.get("title"))
        key = ("arxiv", arxiv_key) if arxiv_key else ("doi", doi_key) if doi_key else ("title", title_key)
        if key in seen:
            continue
        try:
            grade = int(j.get("grade", j.get("relevance", -1)))
        except (TypeError, ValueError):
            continue
        if grade not in (0, 1, 2):
            continue
        seen.add(key)
        valid.append(dict(j, grade=grade))

    output: dict[str, Any] = {
        "relevance_judgment_count": len(valid),
        "relevance_judged_relevant_count": sum(j["grade"] > 0 for j in valid),
    }

    if not valid:
        for k in ks:
            output[f"judged_precision_at_{k}"] = None
            output[f"judged_recall_at_{k}"] = None
            output[f"judged_ndcg_at_{k}"] = None
            output[f"judgment_coverage_at_{k}"] = None
            output[f"judged_results_at_{k}"] = 0
        output.update({"judged_mrr": None, "judged_average_precision": None})
        return output

    def _grade_for(p: Any) -> int | None:
        matched: list[int] = []
        for judgment in valid:
            if match_paper(judgment, p):
                matched.append(judgment["grade"])
        return max(matched) if matched else None

    ranked_grades: list[int | None] = [_grade_for(p) for p in ranked_papers]
    relevant_total = sum(j["grade"] > 0 for j in valid)

    for k in ks:
        if k <= 0:
            raise ValueError("k values must be positive integers")
        top = ranked_grades[:k]
        judged_top = [g for g in top if g is not None]
        retrieved_relevant = sum(g > 0 for g in judged_top)
        denominator = len(judged_top)
        result_count = min(k, len(ranked_papers))
        output[f"judged_results_at_{k}"] = denominator
        output[f"judgment_coverage_at_{k}"] = round(denominator / result_count, 4) if result_count else None
        output[f"judged_precision_at_{k}"] = round(retrieved_relevant / denominator, 4) if denominator else None
        output[f"judged_recall_at_{k}"] = round(retrieved_relevant / relevant_total, 4) if relevant_total else None

        dcg = sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(top) if grade is not None)
        ideal = sorted((j["grade"] for j in valid), reverse=True)[:k]
        idcg = sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
        output[f"judged_ndcg_at_{k}"] = round(dcg / idcg, 4) if idcg else None

    judged_ranked = [grade for grade in ranked_grades if grade is not None]
    first_relevant = next((i for i, grade in enumerate(judged_ranked, 1) if grade > 0), None)
    output["judged_mrr"] = round(1.0 / first_relevant, 4) if first_relevant else (0.0 if relevant_total else None)

    found = 0
    precision_sum = 0.0
    for rank, grade in enumerate(judged_ranked, 1):
        if grade > 0:
            found += 1
            precision_sum += found / rank
    output["judged_average_precision"] = round(precision_sum / relevant_total, 4) if relevant_total else None

    return output


def judged_aspect_coverage(
    aspects: Sequence[Mapping[str, Any]] | None,
    judgments: Sequence[Mapping[str, Any]] | None,
    shortlisted_papers: Sequence[Any],
) -> dict[str, Any]:
    """Evaluates coverage of curated topic aspects by positively judged candidates."""
    aspect_rows = [a for a in (aspects or []) if isinstance(a, Mapping) and a.get("id")]
    judgment_rows = [j for j in (judgments or []) if isinstance(j, Mapping)]
    if not aspect_rows or not judgment_rows:
        return {"judged_aspect_coverage": None, "aspect_coverage": {}}

    per_aspect: dict[str, float | None] = {}
    for aspect in aspect_rows:
        aspect_id = str(aspect["id"])
        has_positive = any(
            int(j.get("grade", j.get("relevance", 0)) or 0) > 0 and aspect_id in (j.get("aspects") or [])
            for j in judgment_rows
        )
        if not has_positive:
            per_aspect[aspect_id] = None
            continue
        covered = any(
            int(j.get("grade", j.get("relevance", 0)) or 0) > 0
            and aspect_id in (j.get("aspects") or [])
            and any(match_paper(j, p) for p in shortlisted_papers)
            for j in judgment_rows
        )
        per_aspect[aspect_id] = 1.0 if covered else 0.0

    evaluable = [v for v in per_aspect.values() if v is not None]
    macro = round(sum(evaluable) / len(evaluable), 4) if evaluable else None
    return {"judged_aspect_coverage": macro, "aspect_coverage": per_aspect}

