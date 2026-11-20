from app.evals.scorers.baseline import NaiveBaselineRunner
from app.evals.scorers.budget import score_budget_adherence
from app.evals.scorers.citation import score_citations
from app.evals.scorers.citation_support import score_citation_support
from app.evals.scorers.contradiction import score_contradiction_detection
from app.evals.scorers.coverage import score_plan_coverage
from app.evals.scorers.dedupe import score_source_diversity
from app.evals.scorers.retrieval_ranking import (
    average_precision,
    coverage,
    graded_relevance_metrics,
    judged_aspect_coverage,
    match_paper,
    mean_average_precision,
    mean_reciprocal_rank,
    mrr,
    ndcg_at_k,
    normalize_arxiv,
    normalize_doi,
    normalize_title,
    precision_at_k,
    recall_at_k,
)

# Backward compatibility aliases
score_budget = score_budget_adherence
score_contradictions = score_contradiction_detection
score_deduplication = score_source_diversity

__all__ = [
    "score_citations",
    "score_citation_support",
    "score_plan_coverage",
    "score_contradictions",
    "score_contradiction_detection",
    "score_deduplication",
    "score_source_diversity",
    "score_budget",
    "score_budget_adherence",
    "NaiveBaselineRunner",
    "normalize_title",
    "normalize_doi",
    "normalize_arxiv",
    "match_paper",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "mean_reciprocal_rank",
    "mrr",
    "average_precision",
    "mean_average_precision",
    "coverage",
    "graded_relevance_metrics",
    "judged_aspect_coverage",
]
