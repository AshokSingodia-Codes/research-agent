// Multi-Agent Research Assistant - Interactive Frontend Application

let currentRunId = null;
let pollTimer = null;
let eventSource = null;

document.addEventListener('DOMContentLoaded', () => {
    // Check API Health on load
    checkHealth();

    // Allow Enter key to trigger research
    const topicInput = document.getElementById('topicInput');
    if (topicInput) {
        topicInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') startResearch();
        });
    }
});

async function checkHealth() {
    try {
        const resp = await fetch('/health');
        if (resp.ok) {
            const data = await resp.json();
            const badge = document.getElementById('engineBadge');
            if (badge) {
                const llmMode = data.mock_llm ? "Mock/Dry-Run" : "Anthropic Claude";
                const redisMode = data.use_fake_redis ? "In-Memory FakeRedis" : "Redis Server";
                badge.innerText = `Engine: ${llmMode} · ${redisMode}`;
            }
        }
    } catch (err) {
        console.warn("Health check error", err);
    }
}

function setPreset(topic) {
    const input = document.getElementById('topicInput');
    if (input) {
        input.value = topic;
        input.focus();
    }
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

    if (tabName === 'report') {
        document.getElementById('tabBtnReport').classList.add('active');
        document.getElementById('reportPane').classList.add('active');
    } else if (tabName === 'trace') {
        document.getElementById('tabBtnTrace').classList.add('active');
        document.getElementById('tracePane').classList.add('active');
        fetchTrace();
    }
}

function updatePipelineHighlight(nodeName, isComplete = false) {
    const nodes = ['planner', 'researcher', 'supervisor', 'writer'];
    nodes.forEach(name => {
        const el = document.getElementById(`node-${name}`);
        if (!el) return;
        el.classList.remove('active');
        if (isComplete) {
            el.classList.add('completed');
        }
    });

    if (!isComplete && nodeName) {
        const activeEl = document.getElementById(`node-${nodeName.toLowerCase()}`);
        if (activeEl) {
            activeEl.classList.add('active');
        }
    }
}

async function startResearch() {
    const topicInput = document.getElementById('topicInput');
    const topic = topicInput ? topicInput.value.trim() : "";
    if (!topic) return;

    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = true;

    // Reset pipeline & results
    updatePipelineHighlight('planner', false);
    const progressBox = document.getElementById('progressBox');
    progressBox.style.display = 'block';
    setProgress(10, 'Initializing multi-agent graph...');

    const resultsCard = document.getElementById('resultsCard');
    resultsCard.style.display = 'block';

    try {
        const resp = await fetch('/research', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: topic })
        });

        if (!resp.ok) {
            throw new Error(`HTTP error ${resp.status}`);
        }

        const data = await resp.json();
        currentRunId = data.run_id;

        const idBadge = document.getElementById('runIdDisplay');
        if (idBadge) idBadge.innerText = `Run ID: ${currentRunId}`;

        // Connect SSE stream with polling fallback
        initStreamOrPolling(currentRunId);

    } catch (err) {
        alert("Failed to start research run: " + err.message);
        runBtn.disabled = false;
        setProgress(0, "Error starting run.");
    }
}

function initStreamOrPolling(runId) {
    if (eventSource) {
        eventSource.close();
    }

    try {
        eventSource = new EventSource(`/research/${runId}/stream`);

        eventSource.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload.type === 'init' || payload.type === 'status') {
                    if (payload.data && payload.data.node) {
                        updatePipelineHighlight(payload.data.node);
                    }
                } else if (payload.type === 'complete' || payload.type === 'stream_end') {
                    eventSource.close();
                    fetchFinalResults(runId);
                }
            } catch (e) {
                // Ignore parse errors on heartbeat pings
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
            // Fall back to polling
            startPolling(runId);
        };
    } catch (e) {
        startPolling(runId);
    }

    // Always maintain background polling for status sync
    startPolling(runId);
}

function startPolling(runId) {
    if (pollTimer) clearInterval(pollTimer);

    pollTimer = setInterval(async () => {
        try {
            const resp = await fetch(`/research/${runId}`);
            if (!resp.ok) return;

            const data = await resp.json();
            const pct = Math.max(data.progress_percentage || 0, 15);
            const currentNode = data.current_node || 'in_progress';

            updatePipelineHighlight(currentNode, data.status === 'completed');
            setProgress(pct, `Agent Stage: ${currentNode.toUpperCase()} (${data.completed_subquestions}/${data.total_subquestions} sub-questions complete)`);

            // Refresh trace periodically
            fetchTrace();

            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(pollTimer);
                if (eventSource) eventSource.close();
                document.getElementById('runBtn').disabled = false;
                setProgress(100, data.status === 'completed' ? 'Research Complete · Citations Grounded' : `Run Failed: ${data.error || 'Unknown error'}`);
                if (data.report) {
                    renderReport(data.report);
                }
            }
        } catch (err) {
            console.warn("Polling error", err);
        }
    }, 1000);
}

async function fetchFinalResults(runId) {
    try {
        const resp = await fetch(`/research/${runId}`);
        if (resp.ok) {
            const data = await resp.json();
            if (data.report) {
                renderReport(data.report);
            }
            updatePipelineHighlight(null, true);
            document.getElementById('runBtn').disabled = false;
            setProgress(100, 'Research Complete · Citations Grounded');
            fetchTrace();
        }
    } catch (e) {
        console.warn("Error fetching final result", e);
    }
}

function setProgress(percent, label) {
    const bar = document.getElementById('progressBar');
    const pctLabel = document.getElementById('progressPercentLabel');
    const stageLabel = document.getElementById('progressStageLabel');

    if (bar) bar.style.width = `${percent}%`;
    if (pctLabel) pctLabel.innerText = `${Math.round(percent)}%`;
    if (stageLabel) stageLabel.innerText = label;
}

function renderReport(report) {
    const container = document.getElementById('reportOutput');
    if (!container) return;

    let html = `
        <div class="report-header">
            <h2>${escapeHtml(report.topic)}</h2>
            <div class="exec-summary">
                <strong>Executive Summary:</strong><br>
                ${escapeHtml(report.executive_summary)}
            </div>
        </div>
    `;

    // Thematic body sections
    if (report.sections && report.sections.length) {
        report.sections.forEach(sec => {
            let sectionContent = escapeHtml(sec.content);

            // Turn citation tags [1], [2] into clickable badges
            sectionContent = sectionContent.replace(/\[(\d+)\]/g, (match) => {
                return `<a href="#citation-${match}" class="citation-badge">${match}</a>`;
            });

            html += `
                <div class="report-body-section">
                    <h3>${escapeHtml(sec.heading)}</h3>
                    <p>${sectionContent}</p>
                </div>
            `;
        });
    }

    // Provenance Ledger
    if (report.citations && report.citations.length) {
        html += `
            <div class="citation-ledger-box">
                <h3 style="font-size: 1.25rem; color: #fff; margin-bottom: 1rem;">
                    Verified Source Citations (${report.citations.length})
                </h3>
        `;

        report.citations.forEach(c => {
            html += `
                <div class="citation-card" id="citation-${escapeHtml(c.citation_id)}">
                    <div class="citation-idx">${escapeHtml(c.citation_id)}</div>
                    <div class="citation-details">
                        <a href="${escapeHtml(c.source_url)}" class="citation-url" target="_blank" rel="noopener noreferrer">
                            ${escapeHtml(c.source_url)}
                        </a>
                        <div class="citation-claim">
                            <strong>Supported Claim:</strong> ${escapeHtml(c.claim_supported)}
                        </div>
                        <div class="citation-snippet">
                            "${escapeHtml(c.snippet_reference)}"
                        </div>
                    </div>
                </div>
            `;
        });

        html += `</div>`;
    }

    // Unresolved questions if any
    if (report.unresolved_questions && report.unresolved_questions.length) {
        html += `
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px;">
                <h4 style="color: var(--amber); margin-bottom: 0.5rem;">Unresolved Sub-Questions (Budget Constrained):</h4>
                <ul style="padding-left: 1.25rem; color: #cbd5e1;">
                    ${report.unresolved_questions.map(q => `<li>${escapeHtml(q)}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    container.innerHTML = html;
}

async function fetchTrace() {
    if (!currentRunId) return;

    try {
        const resp = await fetch(`/research/${currentRunId}/trace`);
        if (!resp.ok) return;

        const data = await resp.json();
        
        // Update metric summary strip
        const stepCountEl = document.getElementById('metricSteps');
        const tokenCountEl = document.getElementById('metricTokens');
        const latencyCountEl = document.getElementById('metricLatency');
        const traceTabBadge = document.getElementById('traceTabCount');

        if (stepCountEl) stepCountEl.innerText = data.total_steps;
        if (tokenCountEl) tokenCountEl.innerText = data.total_tokens.toLocaleString();
        if (latencyCountEl) latencyCountEl.innerText = `${Math.round(data.total_latency_ms)} ms`;
        if (traceTabBadge) traceTabBadge.innerText = data.total_steps;

        const container = document.getElementById('traceList');
        if (!container) return;

        container.innerHTML = data.steps.map(step => {
            const agentPillClass = `pill-${step.node_name.toLowerCase()}`;
            return `
                <div class="trace-step-card">
                    <div class="step-card-header">
                        <div style="display: flex; align-items: center; gap: 0.65rem;">
                            <span class="step-agent-pill ${agentPillClass}">${escapeHtml(step.node_name)}</span>
                            <span style="font-family: var(--font-mono); font-size: 0.85rem; font-weight: 600;">
                                ${escapeHtml(step.step_id)}
                            </span>
                        </div>
                        <div class="step-timing">
                            ${step.latency_ms.toFixed(1)} ms · ${step.tokens_used} tokens
                        </div>
                    </div>
                    <div class="step-io">
                        <strong>Input:</strong> ${escapeHtml(step.input_summary)}
                    </div>
                    <div class="step-io">
                        <strong>Output:</strong> ${escapeHtml(step.output_summary)}
                    </div>
                    ${step.tool_calls && step.tool_calls.length ? `
                        <div class="tool-json-box">
Tool Calls: ${escapeHtml(JSON.stringify(step.tool_calls, null, 2))}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');

    } catch (err) {
        console.warn("Trace fetch error", err);
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
