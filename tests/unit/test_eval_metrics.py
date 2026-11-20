"""Unit tests for pure-Python ranked retrieval and graded relevance evaluation metrics."""

import math
from types import SimpleNamespace

from app.evals.scorers.retrieval_ranking import (
    average_precision,
    coverage,
    graded_relevance_metrics,
    judged_aspect_coverage,
    match_paper,
    mean_reciprocal_rank,
    mrr,
    ndcg_at_k,
    normalize_arxiv,
    normalize_doi,
    normalize_title,
    precision_at_k,
    recall_at_k,
)


def mk_paper(title: str, arxiv: str = "", doi: str = ""):
    return SimpleNamespace(title=title, arxiv_id=arxiv, doi=doi, claim=title)


def test_normalize_title():
    assert normalize_title("Attention Is All You Need!") == "attention is all you need"
    assert normalize_title("Signs as Tokens: A Multi-Lingual Generator") == "signs as tokens a multi lingual generator"


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1007/s11263-019-01281-2") == "10.1007/s11263-019-01281-2"
    assert normalize_doi("doi:10.1145/3308558.3313641.") == "10.1145/3308558.3313641"


def test_normalize_arxiv():
    assert normalize_arxiv("https://arxiv.org/abs/2411.17799v2") == "2411.17799"
    assert normalize_arxiv("arxiv:2404.11499.pdf") == "2404.11499"


def test_match_paper():
    p = mk_paper("Signs as Tokens: Multilingual Sign Generator", arxiv="2411.17799")
    assert match_paper({"arxiv_id": "2411.17799"}, p)
    assert not match_paper({"arxiv_id": "9999.00000"}, p)
    assert match_paper({"title": "signs as tokens multilingual sign generator"}, p)


def test_coverage():
    must = [{"arxiv_id": "0001.1"}, {"arxiv_id": "0002.2"}]
    p1 = mk_paper("P1", arxiv="0001.1")
    p2 = mk_paper("P2", arxiv="0002.2")

    score, missed = coverage(must, [p1, p2])
    assert score == 1.0 and missed == []

    score_part, missed_part = coverage(must, [p1])
    assert score_part == 0.5 and len(missed_part) == 1


def test_precision_and_recall_at_k():
    must = [{"arxiv_id": "0000.2"}, {"arxiv_id": "0000.5"}]
    ranked = [mk_paper(f"Paper {i}", arxiv=f"0000.{i}") for i in range(1, 9)]

    # Top-1: not a must-cite -> P@1 = 0, R@1 = 0
    assert precision_at_k(must, ranked, 1) == 0.0
    assert recall_at_k(must, ranked, 1) == 0.0

    # Top-2: hits rank 2 -> P@2 = 0.5, R@2 = 0.5
    assert precision_at_k(must, ranked, 2) == 0.5
    assert recall_at_k(must, ranked, 2) == 0.5

    # Top-5: hits ranks 2 and 5 -> P@5 = 2/5 = 0.4, R@5 = 2/2 = 1.0
    assert precision_at_k(must, ranked, 5) == 0.4
    assert recall_at_k(must, ranked, 5) == 1.0


def test_ndcg_at_k():
    must = [{"arxiv_id": "0000.2"}, {"arxiv_id": "0000.5"}]
    ranked = [mk_paper(f"Paper {i}", arxiv=f"0000.{i}") for i in range(1, 9)]

    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = round(dcg / idcg, 4)

    assert ndcg_at_k(must, ranked, 5) == expected
    assert ndcg_at_k([], ranked, 5) is None


def test_mrr_and_average_precision():
    must = [{"arxiv_id": "0000.2"}, {"arxiv_id": "0000.5"}]
    ranked = [mk_paper(f"Paper {i}", arxiv=f"0000.{i}") for i in range(1, 9)]

    # First hit at rank 2 -> MRR = 1/2 = 0.5
    assert mean_reciprocal_rank(must, ranked) == 0.5
    assert mrr(must, ranked) == 0.5

    # Hits at rank 2 (P=1/2) and rank 5 (P=2/5) -> AP = (0.5 + 0.4)/2 = 0.45
    assert average_precision(must, ranked) == 0.45


def test_graded_relevance_metrics():
    judgments = [
        {"title": "Directly Relevant Paper", "arxiv_id": "1111.1", "grade": 2},
        {"title": "Partially Relevant Paper", "arxiv_id": "2222.2", "grade": 1},
        {"title": "Irrelevant Paper", "arxiv_id": "3333.3", "grade": 0},
    ]

    p1 = mk_paper("Directly Relevant Paper", arxiv="1111.1")
    p2 = mk_paper("Partially Relevant Paper", arxiv="2222.2")
    p3 = mk_paper("Irrelevant Paper", arxiv="3333.3")

    metrics = graded_relevance_metrics(judgments, [p1, p2, p3], ks=(1, 2, 3))
    assert metrics["relevance_judgment_count"] == 3
    assert metrics["relevance_judged_relevant_count"] == 2
    assert metrics["judged_precision_at_1"] == 1.0
    assert metrics["judged_recall_at_1"] == 0.5
    assert metrics["judged_mrr"] == 1.0


def test_judged_aspect_coverage():
    aspects = [{"id": "syntax", "name": "Grammar syntax"}, {"id": "diffusion", "name": "Diffusion generation"}]
    judgments = [
        {"title": "P1", "arxiv_id": "1111.1", "grade": 2, "aspects": ["syntax"]},
        {"title": "P2", "arxiv_id": "2222.2", "grade": 1, "aspects": ["diffusion"]},
    ]
    p1 = mk_paper("P1", arxiv="1111.1")

    res = judged_aspect_coverage(aspects, judgments, [p1])
    assert res["aspect_coverage"]["syntax"] == 1.0
    assert res["aspect_coverage"]["diffusion"] == 0.0
    assert res["judged_aspect_coverage"] == 0.5

