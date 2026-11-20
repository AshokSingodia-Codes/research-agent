# MOCK: pipeline smoke test, not a quality result

> [!WARNING]
> MOCK: pipeline smoke test, not a quality result. Evaluated using mock providers and synthetic fixtures.

# Research-Agent Benchmark Evaluation Report (MOCK)

> [!NOTE]
> Automated evaluation across 2 benchmark tasks spanning technical deep dives, adversarial edge cases, and contradiction detection.
> Mode: **MOCK** | Cumulative Cost: **$0.0000**

## Summary Metrics & Statistical Lift vs. Naive Baseline

| Evaluation Metric | Multi-Agent System | Naive Baseline | Absolute Lift |
| :--- | :---: | :---: | :---: |
| **Citation Existence / Validity** | **100.0%** | 0.0% | **+100.0%** |
| **Citation Factual Support (Judge)** | **100.0%** | 0.0% | **+100.0%** |
| **Plan-to-Report Coverage** | **25.0%** | 0.0% | **+25.0%** |
| **Domain Diversity Ratio** | **100.0%** | 100.0% | **+0.0%** |
| **Contradiction Detection Accuracy** | **100.0%** | 100.0% | **+0.0%** |
| **Budget & Cost Compliance** | **100.0%** | N/A | **Enforced in Code** |

---

## Per-Query Benchmark Breakdown

| Task ID | Category | Topic | Citation Validity | Citation Support | Plan Coverage | Budget Compliant |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `eval_01` | technical | Raft vs Paxos log compaction under high write... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_02` | technical | Zero-Knowledge rollup throughput bottlenecks ... | 100.0% | 100.0% | 25.0% | Yes |
