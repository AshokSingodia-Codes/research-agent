"""Streamlit Research Dashboard for multi-agent literature and technical research."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.config import settings
from app.graph.builder import build_research_graph
from app.graph.state import AgentState
from app.models.enums import RunStatus

st.set_page_config(
    page_title="Superior Research Agent",
    page_icon="🔬",
    layout="wide",
)

VERDICT_BADGE = {
    "supported": "✅ Supported",
    "partial": "🟡 Partial",
    "unsupported": "❌ Unsupported",
}


def run_research_pipeline(topic: str) -> dict:
    graph = build_research_graph()
    run_id = f"run_ui_{int(time.time())}"
    initial_state: AgentState = {
        "run_id": run_id,
        "question": topic,
        "start_time": time.time(),
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "status": RunStatus.INITIALIZING,
        "findings": [],
        "findings_by_subquestion": {},
        "completed_subquestion_ids": [],
        "completed_branches": [],
        "review_decisions": [],
        "conflicting_evidence": [],
        "trace_log": [],
    }

    final_state = graph.invoke(initial_state)
    return final_state


def render_dashboard(state: dict):
    report = state.get("final_report") or state.get("draft_report")
    critique = state.get("critique")
    findings = state.get("findings", [])
    conflicts = state.get("conflicting_evidence", [])
    traces = state.get("trace_log", [])

    c = st.columns(5)
    c[0].metric("Findings Retrieved", len(findings))
    c[1].metric("Sub-Questions", len(state.get("completed_subquestion_ids", [])))
    c[2].metric("Total Tokens", f"{state.get('total_tokens', 0):,}")
    c[3].metric("Est. USD Cost", f"${state.get('total_usd_cost', 0.0):.4f}")
    cov = getattr(report, "citation_coverage", 1.0) if report else 1.0
    c[4].metric("Citation Coverage", f"{cov:.0%}")

    if critique and not getattr(critique, "passed", True):
        st.warning("Quality Review Note: The Critic flagged areas needing attention. See the Verification tab.")

    report_tab, evidence_tab, verify_tab, contra_tab, trace_tab, export_tab = st.tabs(
        ["Report", "Evidence & Sources", "Claim Verification", "Contradictions", "Agent Trace", "Export"]
    )

    with report_tab:
        if report:
            st.title(report.topic)
            st.subheader("Executive Summary")
            st.write(report.summary)
            for sec in report.sections:
                st.markdown(f"### {sec.title}")
                st.markdown(sec.content)
            if report.citations:
                st.markdown("---")
                st.markdown("### References")
                for k, c_info in sorted(report.citations.items(), key=lambda x: x[0]):
                    st.markdown(f"- **{k}** [{c_info.domain}]({c_info.source_url}) — _{c_info.claim}_")
        else:
            st.info("No report generated yet.")

    with evidence_tab:
        if findings:
            rows = []
            for idx, f in enumerate(findings, 1):
                rows.append({
                    "#": idx,
                    "Domain": f.domain,
                    "Claim": f.claim[:120],
                    "Source URL": f.source_url,
                    "Quality": f"{f.source_quality_score:.2f}",
                    "SubQuestion": f.sub_question_id,
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No evidence findings recorded.")

    with verify_tab:
        if critique and getattr(critique, "unsupported_claims", None):
            st.subheader("Unsupported Claims Flagged by Critic")
            for c_text in critique.unsupported_claims:
                st.error(f"❌ {c_text}")
        else:
            st.success("All cited assertions verified against source excerpts.")

        if critique and getattr(critique, "missing_plan_coverage", None):
            st.subheader("Missing Plan Coverage")
            for m in critique.missing_plan_coverage:
                st.warning(f"⚠️ {m}")

    with contra_tab:
        if conflicts:
            st.subheader("Detected Cross-Source Contradictions")
            for c in conflicts:
                with st.expander(f"Conflict ({c.severity.upper()}): {c.explanation[:80]}..."):
                    st.markdown(f"**Claim A:** {c.claim_a}")
                    st.markdown(f"**Claim B:** {c.claim_b}")
                    st.markdown(f"**Sources:** {', '.join(c.sources)}")
                    st.markdown(f"**Analysis:** {c.explanation}")
        else:
            st.info("No conflicting empirical claims detected across sources.")

    with trace_tab:
        if traces:
            trace_rows = []
            for t in traces:
                trace_rows.append({
                    "Agent": t.agent,
                    "Provider": t.provider,
                    "Model": t.model,
                    "Tokens": t.total_tokens,
                    "Cost ($)": f"${t.usd_cost:.5f}",
                    "Latency (ms)": f"{t.latency_ms:.1f}",
                    "Retries": t.retries,
                    "Error": t.error or "None",
                })
            st.dataframe(trace_rows, use_container_width=True)
        else:
            st.info("No trace steps available.")

    with export_tab:
        if report:
            md_output = f"# {report.topic}\n\n## Executive Summary\n{report.summary}\n\n"
            for sec in report.sections:
                md_output += f"### {sec.title}\n{sec.content}\n\n"
            if report.citations:
                md_output += "### References\n"
                for k, c_info in sorted(report.citations.items(), key=lambda x: x[0]):
                    md_output += f"- **{k}** [{c_info.domain}]({c_info.source_url}) — {c_info.claim}\n"

            st.download_button("Download Report (Markdown)", md_output, "research_report.md", "text/markdown")
            st.download_button("Download Full State (JSON)", json.dumps(state, default=str, indent=2), "run_state.json", "application/json")


def main():
    st.title("🔬 Superior Research Copilot & Literature Engine")
    st.caption("Hybrid Academic (arXiv + OpenAlex) and Web Search with Vector Evidence Indexing, Contradiction Detection, and Grounded Citation Verification.")

    with st.sidebar:
        st.header("Search & Execution Options")
        st.markdown(f"**Active Provider:** `{settings.planner_provider}`")
        st.markdown(f"**Planner Model:** `{settings.planner_model}`")
        st.markdown(f"**Writer Model:** `{settings.writer_model}`")
        st.markdown(f"**Critic Model:** `{settings.critic_model}`")
        academic_on = st.checkbox("Enable Academic Search (arXiv + OpenAlex)", value=True)
        settings.enable_academic_search = academic_on

    topic = st.text_area(
        "Research Question or Topic",
        placeholder="e.g., Cache coherence protocols in distributed GPU memory systems or Zero-knowledge rollup throughput bottlenecks",
        height=100,
    )

    if st.button("Start Deep Research", type="primary"):
        if len(topic.strip()) < 5:
            st.error("Please provide a valid research topic.")
        else:
            with st.spinner("Multi-agent system researching topic across web & academic papers..."):
                state = run_research_pipeline(topic.strip())
                st.session_state["last_state"] = state

    if "last_state" in st.session_state:
        render_dashboard(st.session_state["last_state"])


if __name__ == "__main__":
    main()
