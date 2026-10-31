# Research-Agent Benchmark Evaluation Report (REAL)

> [!NOTE]
> Automated evaluation across 16 benchmark tasks spanning technical deep dives, adversarial edge cases, and contradiction detection.
> Mode: **REAL** | Cumulative Cost: **$0.1023**

## Summary Metrics & Statistical Lift vs. Naive Baseline

| Evaluation Metric | Multi-Agent System | Naive Baseline | Absolute Lift |
| :--- | :---: | :---: | :---: |
| **Citation Existence / Validity** | **87.5%** | 0.0% | **+87.5%** |
| **Citation Factual Support (Judge)** | **68.8%** | 0.0% | **+68.8%** |
| **Plan-to-Report Coverage** | **81.2%** | 0.0% | **+81.2%** |
| **Domain Diversity Ratio** | **90.1%** | 100.0% | **+-9.9%** |
| **Contradiction Detection Accuracy** | **86.2%** | 80.0% | **+6.2%** |
| **Budget & Cost Compliance** | **93.8%** | N/A | **Enforced in Code** |

---

## Per-Query Benchmark Breakdown

| Task ID | Category | Topic | Citation Validity | Citation Support | Plan Coverage | Budget Compliant |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `eval_01` | technical | Raft vs Paxos log compaction under high write... | 100.0% | 100.0% | 100.0% | Yes |
| `eval_02` | technical | Zero-Knowledge rollup throughput bottlenecks ... | 100.0% | 100.0% | 100.0% | Yes |
| `eval_03` | technical | LSM-tree read amplification mitigation via Bl... | 100.0% | 100.0% | 100.0% | Yes |
| `eval_04` | technical | Multi-version concurrency control in distribu... | 100.0% | 100.0% | 100.0% | Yes |
| `eval_05` | technical | eBPF-based packet processing overhead compare... | 100.0% | 100.0% | 100.0% | Yes |
| `eval_06` | technical | Cache coherence protocols in distributed GPU ... | 0.0% | 0.0% | 0.0% | No |
| `eval_07` | adversarial | FlurbleGlock-9000 quantum hypervisor kernel a... | 100.0% | 0.0% | 0.0% | Yes |
| `eval_08` | adversarial | Consensus | 100.0% | 100.0% | 100.0% | Yes |
| `eval_09` | adversarial | Exhaustive historical analysis of every crypt... | 100.0% | 100.0% | 100.0% | Yes |
| `eval_10` | adversarial | SQL vs NoSQL databases in enterprise applicat... | 100.0% | 100.0% | 100.0% | Yes |
| `eval_11` | contradiction | Impact of synchronous multi-datacenter replic... | 100.0% | 100.0% | 100.0% | Yes |
| `eval_12` | contradiction | Optimal garbage collection strategies for hig... | 100.0% | 100.0% | 100.0% | Yes |
| `eval_13` | contradiction | Performance and security trade-offs of kernel... | 100.0% | 100.0% | 100.0% | Yes |
| `eval_14` | architecture | Vector database approximate nearest neighbor ... | 100.0% | 0.0% | 100.0% | Yes |
| `eval_15` | architecture | Microservices vs Modular Monolith latency pro... | 100.0% | 0.0% | 100.0% | Yes |
| `eval_16` | architecture | Distributed tracing instrumentation overhead:... | 0.0% | 0.0% | 0.0% | Yes |
