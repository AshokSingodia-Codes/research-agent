# MOCK: pipeline smoke test, not a quality result

> [!WARNING]
> MOCK: pipeline smoke test, not a quality result. Evaluated using mock providers and synthetic fixtures.

# Research-Agent Benchmark Evaluation Report (MOCK)

> [!NOTE]
> Automated evaluation across 16 benchmark tasks spanning technical deep dives, adversarial edge cases, and contradiction detection.
> Mode: **MOCK** | Cumulative Cost: **$0.0000**

## Summary Metrics & Statistical Lift vs. Naive Baseline

| Evaluation Metric | Multi-Agent System | Naive Baseline | Absolute Lift |
| :--- | :---: | :---: | :---: |
| **Citation Existence / Validity** | **100.0%** | 0.0% | **+100.0%** |
| **Citation Factual Support (Judge)** | **100.0%** | 0.0% | **+100.0%** |
| **Plan-to-Report Coverage** | **25.0%** | 0.0% | **+25.0%** |
| **Domain Diversity Ratio** | **100.0%** | 100.0% | **+0.0%** |
| **Contradiction Detection Accuracy** | **81.2%** | 81.2% | **+0.0%** |
| **Budget & Cost Compliance** | **100.0%** | N/A | **Enforced in Code** |

---

## Per-Query Benchmark Breakdown

| Task ID | Category | Topic | Citation Validity | Citation Support | Plan Coverage | Budget Compliant |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `eval_01` | technical | Raft vs Paxos log compaction under high write... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_02` | technical | Zero-Knowledge rollup throughput bottlenecks ... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_03` | technical | LSM-tree read amplification mitigation via Bl... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_04` | technical | Multi-version concurrency control in distribu... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_05` | technical | eBPF-based packet processing overhead compare... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_06` | technical | Cache coherence protocols in distributed GPU ... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_07` | adversarial | FlurbleGlock-9000 quantum hypervisor kernel a... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_08` | adversarial | Consensus | 100.0% | 100.0% | 25.0% | Yes |
| `eval_09` | adversarial | Exhaustive historical analysis of every crypt... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_10` | adversarial | SQL vs NoSQL databases in enterprise applicat... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_11` | contradiction | Impact of synchronous multi-datacenter replic... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_12` | contradiction | Optimal garbage collection strategies for hig... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_13` | contradiction | Performance and security trade-offs of kernel... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_14` | architecture | Vector database approximate nearest neighbor ... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_15` | architecture | Microservices vs Modular Monolith latency pro... | 100.0% | 100.0% | 25.0% | Yes |
| `eval_16` | architecture | Distributed tracing instrumentation overhead:... | 100.0% | 100.0% | 25.0% | Yes |
