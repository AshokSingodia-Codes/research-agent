"""Report rendering service: converts structured Report schemas to Markdown and interactive HTML."""

from __future__ import annotations

import html
import re
from typing import Any

from app.models.schemas import Report


def _coerce_report(report: Report | dict[str, Any]) -> dict[str, Any]:
    if hasattr(report, "model_dump"):
        return report.model_dump()
    return dict(report)


def render_markdown(report: Report | dict[str, Any]) -> str:
    """Converts a Report into cleanly formatted, publication-ready Markdown."""
    rep = _coerce_report(report)
    topic = rep.get("topic", "Research Report")
    summary = rep.get("summary", "")
    sections = rep.get("sections", [])
    contradictions = rep.get("conflicting_evidence", [])
    citations = rep.get("citations", {})

    lines: list[str] = [
        f"# {topic}",
        "",
        "## Executive Summary",
        summary or "No summary available.",
        "",
    ]

    for section in sections:
        title = section.get("title", "Section")
        content = section.get("content", "")
        lines.append(f"## {title}")
        lines.append(content)
        lines.append("")

    if contradictions:
        lines.append("## Conflicting Evidence & Contradictions")
        for idx, contra in enumerate(contradictions, 1):
            sev = contra.get("severity", "medium").upper()
            expl = contra.get("explanation", "")
            ca = contra.get("claim_a", "")
            cb = contra.get("claim_b", "")
            lines.append(f"### Dispute #{idx} [{sev}]")
            lines.append(f"- **Finding A:** {ca}")
            lines.append(f"- **Finding B:** {cb}")
            lines.append(f"- **Analysis:** {expl}")
            lines.append("")

    if citations:
        lines.append("## References & Evidence Sources")
        for key, cite in sorted(citations.items(), key=lambda x: x[0]):
            claim = cite.get("claim", "")
            url = cite.get("source_url", "")
            domain = cite.get("domain", "")
            lines.append(f"- **{key}** [{domain}]({url}): {claim}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_html(report: Report | dict[str, Any], run_id: str | None = None) -> str:
    """Converts a Report into a modern, responsive HTML document with interactive citation jump-links."""
    rep = _coerce_report(report)
    topic = html.escape(rep.get("topic", "Research Report"))
    summary = html.escape(rep.get("summary", "")).replace("\n", "<br/>")
    sections = rep.get("sections", [])
    contradictions = rep.get("conflicting_evidence", [])
    citations = rep.get("citations", {})
    coverage_pct = round(float(rep.get("citation_coverage", 1.0)) * 100, 1)

    # Linkify inline citations [1], [2] to #cite-1
    def _linkify_citations(text: str) -> str:
        escaped = html.escape(text)

        def _sub_cite(match: re.Match) -> str:
            num = match.group(1)
            return f'<a href="#cite-{num}" class="cite-tag" title="Jump to reference [{num}]">[{num}]</a>'

        return re.sub(r"\[(\d+)\]", _sub_cite, escaped).replace("\n", "<br/>")

    sections_html = []
    for s_idx, sec in enumerate(sections, 1):
        s_title = html.escape(sec.get("title", f"Section {s_idx}"))
        s_content = _linkify_citations(sec.get("content", ""))
        sections_html.append(f"""
        <section class="report-section" id="section-{s_idx}">
            <h2>{s_title}</h2>
            <div class="section-body">{s_content}</div>
        </section>
        """)

    contradictions_html = ""
    if contradictions:
        contra_items = []
        for c_idx, contra in enumerate(contradictions, 1):
            sev = html.escape(str(contra.get("severity", "medium")).lower())
            expl = html.escape(contra.get("explanation", ""))
            ca = html.escape(contra.get("claim_a", ""))
            cb = html.escape(contra.get("claim_b", ""))
            contra_items.append(f"""
            <div class="contradiction-card sev-{sev}">
                <div class="contra-header">
                    <span class="contra-badge">{sev.upper()} SEVERITY</span>
                    <strong>Dispute #{c_idx}</strong>
                </div>
                <div class="contra-claims">
                    <p><strong>Finding A:</strong> {ca}</p>
                    <p><strong>Finding B:</strong> {cb}</p>
                </div>
                <p class="contra-expl"><em>Synthesis:</em> {expl}</p>
            </div>
            """)
        contradictions_html = f"""
        <section class="contradictions-section">
            <h2>Conflicting Evidence & Divergent Findings</h2>
            {"".join(contra_items)}
        </section>
        """

    citations_html = ""
    if citations:
        cite_items = []
        # Sort citation keys numerically
        def _key_sorter(k: str) -> int:
            m = re.search(r"\d+", k)
            return int(m.group(0)) if m else 999

        for key, cite in sorted(citations.items(), key=lambda x: _key_sorter(x[0])):
            m = re.search(r"\d+", key)
            cite_num = m.group(0) if m else key.strip("[]")
            claim = html.escape(cite.get("claim", ""))
            url = html.escape(cite.get("source_url", "#"))
            domain = html.escape(cite.get("domain", "external"))
            cite_items.append(f"""
            <li id="cite-{cite_num}" class="reference-item">
                <span class="ref-marker">[{cite_num}]</span>
                <span class="ref-domain">{domain}</span>
                <a href="{url}" target="_blank" rel="noopener noreferrer" class="ref-link">{url}</a>
                <p class="ref-claim">{claim}</p>
            </li>
            """)
        citations_html = f"""
        <section class="references-section" id="references">
            <h2>References & Cited Evidence</h2>
            <ol class="references-list">
                {"".join(cite_items)}
            </ol>
        </section>
        """

    run_meta = f'<span class="meta-item">Run ID: <code>{html.escape(run_id)}</code></span>' if run_id else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{topic} - Research Report</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.15);
            --badge-bg: #0369a1;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background: var(--bg);
            color: var(--text-main);
            font-family: var(--font-family);
            line-height: 1.65;
            margin: 0;
            padding: 2rem 1rem;
        }}
        .report-container {{
            max-width: 900px;
            margin: 0 auto;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}
        h1 {{
            font-size: 2.1rem;
            margin-top: 0;
            margin-bottom: 0.75rem;
            color: #ffffff;
            letter-spacing: -0.02em;
        }}
        .meta-bar {{
            display: flex;
            gap: 1.5rem;
            font-size: 0.85rem;
            color: var(--text-muted);
            flex-wrap: wrap;
        }}
        .meta-item code {{
            background: #0f172a;
            padding: 2px 6px;
            border-radius: 4px;
            color: var(--accent);
        }}
        .summary-box {{
            background: rgba(56, 189, 248, 0.05);
            border-left: 4px solid var(--accent);
            padding: 1.25rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 2.5rem;
        }}
        .summary-box h2 {{
            margin-top: 0;
            font-size: 1.2rem;
            color: var(--accent);
        }}
        .report-section {{
            margin-bottom: 2.5rem;
        }}
        .report-section h2 {{
            font-size: 1.4rem;
            color: #ffffff;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding-bottom: 0.5rem;
        }}
        .section-body {{
            color: #e2e8f0;
            font-size: 1.02rem;
        }}
        .cite-tag {{
            display: inline-block;
            background: var(--badge-bg);
            color: #ffffff;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 1px 5px;
            border-radius: 4px;
            text-decoration: none;
            vertical-align: super;
            margin: 0 2px;
            transition: all 0.2s ease;
        }}
        .cite-tag:hover {{
            background: #0284c7;
            transform: translateY(-2px);
            box-shadow: 0 2px 8px var(--accent-glow);
        }}
        .contradiction-card {{
            background: #181f30;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            border-left: 4px solid #f59e0b;
        }}
        .contradiction-card.sev-high {{
            border-left-color: #ef4444;
        }}
        .contra-badge {{
            font-size: 0.7rem;
            padding: 2px 6px;
            background: #f59e0b;
            color: #000000;
            font-weight: bold;
            border-radius: 4px;
            margin-right: 0.5rem;
        }}
        .contradiction-card.sev-high .contra-badge {{
            background: #ef4444;
            color: #ffffff;
        }}
        .contra-claims p {{
            margin: 0.4rem 0;
            font-size: 0.95rem;
        }}
        .references-section {{
            border-top: 2px solid var(--border);
            padding-top: 2rem;
            margin-top: 3rem;
        }}
        .references-list {{
            list-style: none;
            padding: 0;
        }}
        .reference-item {{
            background: #141d2d;
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.75rem;
            transition: border-color 0.2s ease;
        }}
        .reference-item:target {{
            border-color: var(--accent);
            box-shadow: 0 0 12px var(--accent-glow);
        }}
        .ref-marker {{
            font-weight: bold;
            color: var(--accent);
            margin-right: 0.5rem;
        }}
        .ref-domain {{
            font-size: 0.75rem;
            background: #334155;
            color: #cbd5e1;
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 0.75rem;
        }}
        .ref-link {{
            color: var(--accent);
            font-size: 0.9rem;
            text-decoration: none;
            word-break: break-all;
        }}
        .ref-link:hover {{
            text-decoration: underline;
        }}
        .ref-claim {{
            margin: 0.5rem 0 0 0;
            font-size: 0.92rem;
            color: #cbd5e1;
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <header>
            <h1>{topic}</h1>
            <div class="meta-bar">
                {run_meta}
                <span class="meta-item">Citation Coverage: <strong>{coverage_pct}%</strong></span>
                <span class="meta-item">Sections: <strong>{len(sections)}</strong></span>
                <span class="meta-item">Sources: <strong>{len(citations)}</strong></span>
            </div>
        </header>

        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p>{summary}</p>
        </div>

        {"".join(sections_html)}

        {contradictions_html}

        {citations_html}
    </div>
</body>
</html>
"""
