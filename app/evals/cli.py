import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.evals.scorers.baseline import NaiveBaselineRunner
from app.evals.scorers.budget import score_budget_adherence
from app.evals.scorers.citation import score_citations
from app.evals.scorers.contradiction import score_contradiction_detection
from app.evals.scorers.coverage import score_plan_coverage
from app.evals.scorers.dedupe import score_source_diversity
from app.graph.builder import build_research_graph
from app.llm.client import llm_client
from app.llm.fake_provider import FakeProvider
from app.persistence.checkpointer import RedisCheckpointSaver

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    items = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_evaluations(
    dataset_path: Path,
    output_dir: Path,
    mock: bool = True,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Runs full benchmark evaluation across the dataset and compares against naive baseline."""
    if mock:
        os.environ["MOCK_LLM"] = "true"
        os.environ["MOCK_SEARCH"] = "true"
        settings.mock_llm = "true"
        settings.mock_search = "true"
        fake = FakeProvider()
        fake.artificial_delay_s = 0.0
        llm_client.set_override_provider(fake)

    items = load_dataset(dataset_path)
    if sample_limit:
        items = items[:sample_limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    baseline_runner = NaiveBaselineRunner()

    checkpointer = RedisCheckpointSaver()
    graph = build_research_graph(checkpointer=checkpointer)

    for idx, item in enumerate(items, 1):
        topic = item["topic"]
        eval_id = item.get("id", f"eval_{idx}")
        category = item.get("category", "general")
        contra_expected = item.get("contradiction_expected", False)
        print(f"[{idx}/{len(items)}] Evaluating: {eval_id} ({category}) - '{topic[:40]}...'")

        # 1. Run Multi-Agent System
        t0 = time.time()
        run_id = f"eval_{eval_id}_{int(t0)}"
        state_input = {
            "run_id": run_id,
            "question": topic,
            "start_time": t0,
            "total_tokens": 0,
            "total_usd_cost": 0.0,
            "step_index": 0,
        }
        config = {"configurable": {"thread_id": run_id}}
        final_state = graph.invoke(state_input, config=config)
        agent_latency = time.time() - t0

        final_report = final_state.get("final_report")
        findings = final_state.get("findings", [])
        plan = final_state.get("plan")
        tokens = final_state.get("total_tokens", 0)
        cost = final_state.get("total_usd_cost", 0.0)

        agent_scores = {
            "citation": score_citations(final_report, findings),
            "coverage": score_plan_coverage(plan, final_report),
            "diversity": score_source_diversity(findings),
            "budget": score_budget_adherence(tokens, cost, agent_latency, getattr(final_report, "partial", False)),
            "contradiction": score_contradiction_detection(final_report, contra_expected),
        }

        # 2. Run Naive Baseline
        base_res = baseline_runner.run(topic)
        base_scores = {
            "citation": score_citations(base_res["report"], base_res["verified_findings"]),
            "coverage": score_plan_coverage(None, base_res["report"]),
            "diversity": score_source_diversity(base_res["verified_findings"]),
            "budget": score_budget_adherence(base_res["total_tokens"], base_res["total_usd_cost"], base_res["wall_clock_seconds"]),
            "contradiction": score_contradiction_detection(base_res["report"], contra_expected),
        }

        results.append({
            "id": eval_id,
            "category": category,
            "topic": topic,
            "agent": agent_scores,
            "baseline": base_scores,
        })

    # Compute Summary Aggregations
    summary = aggregate_benchmark_results(results)

    # Save JSON and Markdown artifacts
    json_path = output_dir / "eval_report.json"
    md_path = output_dir / "eval_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    markdown_report = generate_markdown_report(summary, results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    print("\n[+] Benchmark completed. Artifacts saved:")
    print(f"    - JSON: {json_path}")
    print(f"    - Markdown: {md_path}")
    return {"summary": summary, "results": results}


def aggregate_benchmark_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    avg_agent_citation_val = sum(r["agent"]["citation"]["citation_validity_ratio"] for r in results) / n
    avg_base_citation_val = sum(r["baseline"]["citation"]["citation_validity_ratio"] for r in results) / n

    avg_agent_coverage = sum(r["agent"]["coverage"]["plan_coverage_ratio"] for r in results) / n
    avg_base_coverage = sum(r["baseline"]["coverage"]["plan_coverage_ratio"] for r in results) / n

    avg_agent_diversity = sum(r["agent"]["diversity"]["domain_diversity_ratio"] for r in results) / n
    avg_base_diversity = sum(r["baseline"]["diversity"]["domain_diversity_ratio"] for r in results) / n

    agent_contra_accuracy = sum(r["agent"]["contradiction"]["contradiction_accuracy"] for r in results) / n
    base_contra_accuracy = sum(r["baseline"]["contradiction"]["contradiction_accuracy"] for r in results) / n

    budget_compliance = sum(1 for r in results if r["agent"]["budget"]["budget_compliance"]) / n

    return {
        "total_evaluations": n,
        "agent": {
            "mean_citation_validity": round(avg_agent_citation_val, 4),
            "mean_plan_coverage": round(avg_agent_coverage, 4),
            "mean_domain_diversity": round(avg_agent_diversity, 4),
            "contradiction_accuracy": round(agent_contra_accuracy, 4),
            "budget_compliance_ratio": round(budget_compliance, 4),
        },
        "baseline": {
            "mean_citation_validity": round(avg_base_citation_val, 4),
            "mean_plan_coverage": round(avg_base_coverage, 4),
            "mean_domain_diversity": round(avg_base_diversity, 4),
            "contradiction_accuracy": round(base_contra_accuracy, 4),
        },
        "statistical_lift": {
            "citation_validity_lift": round(avg_agent_citation_val - avg_base_citation_val, 4),
            "plan_coverage_lift": round(avg_agent_coverage - avg_base_coverage, 4),
            "contradiction_lift": round(agent_contra_accuracy - base_contra_accuracy, 4),
        },
    }


def generate_markdown_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    agent = summary.get("agent", {})
    base = summary.get("baseline", {})
    lift = summary.get("statistical_lift", {})

    lines = [
        "# Research-Agent Benchmark Evaluation Report",
        "",
        "> [!NOTE]",
        f"> Automated evaluation across {summary.get('total_evaluations', 0)} benchmark tasks spanning technical deep dives, adversarial edge cases, and contradiction detection.",
        "",
        "## Summary Metrics & Statistical Lift vs. Naive Baseline",
        "",
        "| Evaluation Metric | Multi-Agent System | Naive Baseline | Absolute Lift |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Citation Grounding / Validity** | **{agent.get('mean_citation_validity', 0.0):.1%}** | {base.get('mean_citation_validity', 0.0):.1%} | **+{lift.get('citation_validity_lift', 0.0):.1%}** |",
        f"| **Plan-to-Report Coverage** | **{agent.get('mean_plan_coverage', 0.0):.1%}** | {base.get('mean_plan_coverage', 0.0):.1%} | **+{lift.get('plan_coverage_lift', 0.0):.1%}** |",
        f"| **Domain Diversity Ratio** | **{agent.get('mean_domain_diversity', 0.0):.1%}** | {base.get('mean_domain_diversity', 0.0):.1%} | **+{agent.get('mean_domain_diversity', 0.0) - base.get('mean_domain_diversity', 0.0):.1%}** |",
        f"| **Contradiction Detection Accuracy** | **{agent.get('contradiction_accuracy', 0.0):.1%}** | {base.get('contradiction_accuracy', 0.0):.1%} | **+{lift.get('contradiction_lift', 0.0):.1%}** |",
        f"| **Budget & Cost Compliance** | **{agent.get('budget_compliance_ratio', 0.0):.1%}** | N/A | **Enforced in Code** |",
        "",
        "---",
        "",
        "## Per-Query Benchmark Breakdown",
        "",
        "| Task ID | Category | Topic | Citation Validity | Plan Coverage | Budget Compliant |",
        "| :--- | :--- | :--- | :---: | :---: | :---: |",
    ]

    for r in results:
        eid = r["id"]
        cat = r["category"]
        top = r["topic"][:45] + ("..." if len(r["topic"]) > 45 else "")
        c_val = r["agent"]["citation"]["citation_validity_ratio"]
        p_cov = r["agent"]["coverage"]["plan_coverage_ratio"]
        b_comp = "Yes" if r["agent"]["budget"]["budget_compliance"] else "No"
        lines.append(f"| `{eid}` | {cat} | {top} | {c_val:.1%} | {p_cov:.1%} | {b_comp} |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run research-agent benchmark evaluations.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run benchmark suite")
    run_parser.add_argument("--dataset", type=str, default="app/evals/datasets/core.jsonl", help="Path to jsonl dataset")
    run_parser.add_argument("--output", type=str, default="docs", help="Directory to save report artifacts")
    run_parser.add_argument("--mock", action="store_true", default=True, help="Use mock provider for deterministic evaluation")
    run_parser.add_argument("--limit", type=int, default=None, help="Sample limit")

    args = parser.parse_args()
    if args.command == "run":
        run_evaluations(
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output),
            mock=args.mock,
            sample_limit=args.limit,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
