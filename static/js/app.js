// Research Assistant - Minimalist Calm Frontend

// Support remote backend (e.g. Render) when deployed on Vercel
const API_BASE = window.API_BASE_URL || localStorage.getItem('multiagent_api_base') || '';

function getApiUrl(path) {
    if (!path.startsWith('/')) path = '/' + path;
    return API_BASE ? `${API_BASE.replace(/\/+$/, '')}${path}` : path;
}

let currentRunId = null;
let pollTimer = null;
let eventSource = null;
let apiKey = localStorage.getItem('multiagent_api_key') || 'test-key-12345';

document.addEventListener('DOMContentLoaded', () => {
    checkHealth();

    const topicInput = document.getElementById('topicInput');
    if (topicInput) {
        topicInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                startResearch();
            }
        });
    }
});

async function checkHealth() {
    try {
        const resp = await fetch(getApiUrl('/health'));
        if (resp.ok) {
            const data = await resp.json();
            if (data.default_api_key && !localStorage.getItem('multiagent_api_key')) {
                apiKey = data.default_api_key;
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

function updateStepStatus(stepNumber, state) {
    const stepEl = document.getElementById(`step-${stepNumber}`);
    if (!stepEl) return;

    stepEl.classList.remove('active', 'done');
    const indicator = stepEl.querySelector('.step-indicator');

    if (state === 'active') {
        stepEl.classList.add('active');
        if (indicator) {
            indicator.innerHTML = '<span class="step-dot-active"></span>';
        }
    } else if (state === 'done') {
        stepEl.classList.add('done');
        if (indicator) {
            indicator.innerHTML = `
                <svg class="step-icon-done" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
            `;
        }
    } else {
        if (indicator) {
            indicator.innerHTML = '<span class="step-dot-pending"></span>';
        }
    }
}

function mapNodeToStep(nodeName, isComplete = false) {
    const progressBar = document.getElementById('progressBar');

    if (isComplete) {
        for (let i = 1; i <= 4; i++) {
            updateStepStatus(i, 'done');
        }
        if (progressBar) progressBar.style.width = '100%';
        return;
    }

    const node = (nodeName || '').toLowerCase();

    const statusTextEl = document.getElementById('activityStatusText');
    const subtextEl = document.getElementById('activitySubtext');

    // 1: Understanding your question (planner)
    // 2: Searching trusted sources (researcher)
    // 3: Checking quality (supervisor/critic)
    // 4: Writing your report (writer)
    if (node.includes('planner') || node === 'planning' || node === 'queued') {
        updateStepStatus(1, 'active');
        updateStepStatus(2, 'pending');
        updateStepStatus(3, 'pending');
        updateStepStatus(4, 'pending');
        if (progressBar) progressBar.style.width = '20%';
        if (statusTextEl) statusTextEl.innerText = "Analyzing question & decomposing sub-topics...";
        if (subtextEl) subtextEl.innerText = "Planner is formulating structured research objectives";
    } else if (node.includes('researcher') || node.includes('search')) {
        updateStepStatus(1, 'done');
        updateStepStatus(2, 'active');
        updateStepStatus(3, 'pending');
        updateStepStatus(4, 'pending');
        if (progressBar) progressBar.style.width = '50%';
        if (statusTextEl) statusTextEl.innerText = "Searching live sources via Tavily...";
        if (subtextEl) subtextEl.innerText = "Extracting factual claims and filtering domain quality";
    } else if (node.includes('supervisor') || node.includes('review') || node.includes('critic')) {
        updateStepStatus(1, 'done');
        updateStepStatus(2, 'done');
        updateStepStatus(3, 'active');
        updateStepStatus(4, 'pending');
        if (progressBar) progressBar.style.width = '75%';
        if (statusTextEl) statusTextEl.innerText = "Evaluating evidence & cross-checking contradictions...";
        if (subtextEl) subtextEl.innerText = "Supervisor is validating source quality and citation grounding";
    } else if (node.includes('writer') || node.includes('synthesis')) {
        updateStepStatus(1, 'done');
        updateStepStatus(2, 'done');
        updateStepStatus(3, 'done');
        updateStepStatus(4, 'active');
        if (progressBar) progressBar.style.width = '90%';
        if (statusTextEl) statusTextEl.innerText = "Writing synthesis report & formatting bibliography...";
        if (subtextEl) subtextEl.innerText = "Compiling verified claims and linking footnotes";
    }
}

async function startResearch() {
    const topicInput = document.getElementById('topicInput');
    const topic = topicInput ? topicInput.value.trim() : "";
    if (!topic) return;

    const runBtn = document.getElementById('runBtn');
    const progressBox = document.getElementById('progressBox');
    const reportContainer = document.getElementById('reportContainer');
    const errorBanner = document.getElementById('errorBanner');
    const suggestionsRow = document.getElementById('suggestionsRow');

    // UI state transitions
    if (runBtn) runBtn.disabled = true;
    if (topicInput) topicInput.disabled = true;
    if (errorBanner) errorBanner.style.display = 'none';
    if (reportContainer) reportContainer.style.display = 'none';
    if (suggestionsRow) suggestionsRow.style.display = 'none';
    if (progressBox) progressBox.style.display = 'block';

    mapNodeToStep('planner', false);

    try {
        const resp = await fetch(getApiUrl('/research'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKey
            },
            body: JSON.stringify({ question: topic })
        });

        if (!resp.ok) {
            throw new Error(`HTTP error ${resp.status}`);
        }

        const data = await resp.json();
        currentRunId = data.run_id;

        const detailsId = document.getElementById('detailsRunId');
        if (detailsId) detailsId.innerText = currentRunId;

        initStreamOrPolling(currentRunId);

    } catch (err) {
        showError("Something went wrong. Please try again.", err.message);
        if (runBtn) runBtn.disabled = false;
        if (topicInput) topicInput.disabled = false;
        if (progressBox) progressBox.style.display = 'none';
        if (suggestionsRow) suggestionsRow.style.display = 'flex';
    }
}

function initStreamOrPolling(runId) {
    if (eventSource) {
        eventSource.close();
    }

    try {
        eventSource = new EventSource(getApiUrl(`/research/${runId}/stream?api_key=${encodeURIComponent(apiKey)}`));

        eventSource.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload.type === 'init' || payload.type === 'status') {
                    if (payload.data && payload.data.node) {
                        mapNodeToStep(payload.data.node);
                    }
                } else if (payload.type === 'complete' || payload.type === 'stream_end') {
                    eventSource.close();
                    fetchFinalResults(runId);
                } else if (payload.type === 'cancelled' || payload.event === 'cancelled') {
                    eventSource.close();
                    if (pollTimer) clearInterval(pollTimer);
                    resetInputState();
                }
            } catch (e) {
                // Ignore parse errors on heartbeat
            }
        };

        eventSource.onerror = () => {
            if (eventSource) eventSource.close();
            startPolling(runId);
        };
    } catch (e) {
        startPolling(runId);
    }

    startPolling(runId);
}

function startPolling(runId) {
    if (pollTimer) clearInterval(pollTimer);

    pollTimer = setInterval(async () => {
        try {
            const resp = await fetch(getApiUrl(`/research/${runId}`), {
                headers: { 'X-API-Key': apiKey }
            });
            if (!resp.ok) return;

            const data = await resp.json();
            const currentNode = data.current_node || (data.status === 'queued' ? 'planning' : 'researcher');

            mapNodeToStep(currentNode, data.status === 'completed');

            // Update details disclosure metrics quietly
            const tokensEl = document.getElementById('detailsTokens');
            const costEl = document.getElementById('detailsCost');
            if (tokensEl && data.total_tokens) tokensEl.innerText = data.total_tokens.toLocaleString();
            if (costEl && data.total_cost_usd) costEl.innerText = `$${data.total_cost_usd.toFixed(4)}`;

            if (Math.random() < 0.25) {
                fetchTraceQuietly();
            }

            if (data.status === 'completed') {
                clearInterval(pollTimer);
                if (eventSource) eventSource.close();
                fetchTraceQuietly();
                onResearchComplete(data.report);
            } else if (data.status === 'cancelled') {
                clearInterval(pollTimer);
                if (eventSource) eventSource.close();
                resetInputState();
            } else if (data.status === 'failed') {
                clearInterval(pollTimer);
                if (eventSource) eventSource.close();
                showError("Something went wrong while preparing your report. Please try again.", data.error);
                resetInputState();
            }
        } catch (err) {
            console.warn("Polling error", err);
        }
    }, 1500);
}

async function cancelResearch() {
    if (!currentRunId) return;

    const cancelBtn = document.getElementById('cancelBtn');
    if (cancelBtn) {
        cancelBtn.disabled = true;
        cancelBtn.innerText = "Cancelling...";
    }

    if (pollTimer) clearInterval(pollTimer);
    if (eventSource) eventSource.close();

    try {
        await fetch(getApiUrl(`/research/${currentRunId}/cancel`), {
            method: 'POST',
            headers: { 'X-API-Key': apiKey }
        });
    } catch (e) {
        // quiet ignore
    } finally {
        currentRunId = null;
        if (cancelBtn) {
            cancelBtn.disabled = false;
            cancelBtn.innerText = "Cancel";
        }
        resetInputState();
    }
}

function resetInputState() {
    const runBtn = document.getElementById('runBtn');
    const topicInput = document.getElementById('topicInput');
    const progressBox = document.getElementById('progressBox');
    const suggestionsRow = document.getElementById('suggestionsRow');

    if (runBtn) runBtn.disabled = false;
    if (topicInput) topicInput.disabled = false;
    if (progressBox) progressBox.style.display = 'none';
    if (suggestionsRow) suggestionsRow.style.display = 'flex';
}

function onResearchComplete(report) {
    resetInputState();
    mapNodeToStep(null, true);

    const progressBox = document.getElementById('progressBox');
    if (progressBox) progressBox.style.display = 'none';

    if (report) {
        renderReport(report);
    }
}

async function fetchFinalResults(runId) {
    try {
        const resp = await fetch(getApiUrl(`/research/${runId}`), {
            headers: { 'X-API-Key': apiKey }
        });
        if (resp.ok) {
            const data = await resp.json();
            if (data.report) {
                onResearchComplete(data.report);
            }
        }
    } catch (e) {
        console.warn("Error fetching final result", e);
    }
}

function renderReport(report) {
    const reportContainer = document.getElementById('reportContainer');
    const reportTitle = document.getElementById('reportTitle');
    const reportSummary = document.getElementById('reportSummary');
    const reportSections = document.getElementById('reportSections');
    const sourcesList = document.getElementById('sourcesList');
    const sourcesSection = document.getElementById('sourcesSection');

    if (!reportContainer) return;

    if (reportTitle) reportTitle.innerText = report.topic || 'Research Findings';
    const summaryText = report.summary || report.executive_summary || '';
    if (reportSummary) {
        if (summaryText.trim()) {
            reportSummary.innerText = summaryText;
            reportSummary.style.display = 'block';
        } else {
            reportSummary.style.display = 'none';
        }
    }

    // Sections
    let sectionsHtml = '';
    if (report.sections && report.sections.length) {
        report.sections.forEach(sec => {
            let sectionContent = escapeHtml(sec.content || '');
            // Convert citation references [1], [2] into clean inline links
            sectionContent = sectionContent.replace(/\[(\d+)\]/g, (match, p1) => {
                return `<a href="#source-${p1}" class="citation-ref">${p1}</a>`;
            });

            const headingTitle = sec.title || sec.heading || '';

            sectionsHtml += `
                <section class="report-section">
                    <h3>${escapeHtml(headingTitle)}</h3>
                    <p>${sectionContent}</p>
                </section>
            `;
        });
    }
    if (reportSections) reportSections.innerHTML = sectionsHtml;

    // Sources: handles both array or dict formats
    let citationsList = [];
    if (Array.isArray(report.citations)) {
        citationsList = report.citations;
    } else if (report.citations && typeof report.citations === 'object') {
        citationsList = Object.entries(report.citations).map(([key, val]) => ({
            citation_id: key.replace(/[\[\]]/g, ''),
            source_url: val.source_url || val.url || '',
            claim_supported: val.claim || val.claim_supported || '',
            domain: val.domain || ''
        }));
    }

    if (citationsList.length) {
        if (sourcesSection) sourcesSection.style.display = 'block';
        if (sourcesList) {
            sourcesList.innerHTML = citationsList.map(c => `
                <li class="source-item" id="source-${escapeHtml(c.citation_id)}">
                    <span class="source-number">[${escapeHtml(c.citation_id)}]</span>
                    <div class="source-content">
                        <a href="${escapeHtml(c.source_url)}" class="source-link" target="_blank" rel="noopener noreferrer">
                            ${escapeHtml(c.source_url)}
                        </a>
                        ${c.claim_supported ? `<div class="source-claim">${escapeHtml(c.claim_supported)}</div>` : ''}
                    </div>
                </li>
            `).join('');
        }
    } else {
        if (sourcesSection) sourcesSection.style.display = 'none';
    }

    reportContainer.style.display = 'block';
}

function downloadMarkdown() {
    const title = document.getElementById('reportTitle')?.innerText || 'Research Report';
    const summary = document.getElementById('reportSummary')?.innerText || '';
    const sections = Array.from(document.querySelectorAll('.report-section')).map(sec => {
        const h = sec.querySelector('h3')?.innerText || '';
        const p = sec.querySelector('p')?.innerText || '';
        return `## ${h}\n\n${p}`;
    }).join('\n\n');

    const sources = Array.from(document.querySelectorAll('.source-item')).map(item => {
        const num = item.querySelector('.source-number')?.innerText || '';
        const link = item.querySelector('.source-link')?.href || '';
        const claim = item.querySelector('.source-claim')?.innerText || '';
        return `${num} ${link}${claim ? ` - ${claim}` : ''}`;
    }).join('\n');

    const fullText = `# ${title}\n\n${summary}\n\n${sections}\n\n### Sources & Bibliography\n\n${sources}\n`;

    const blob = new Blob([fullText], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '_')}_report.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function copyReport() {
    const title = document.getElementById('reportTitle')?.innerText || '';
    const summary = document.getElementById('reportSummary')?.innerText || '';
    const sections = Array.from(document.querySelectorAll('.report-section')).map(sec => {
        const h = sec.querySelector('h3')?.innerText || '';
        const p = sec.querySelector('p')?.innerText || '';
        return `## ${h}\n\n${p}`;
    }).join('\n\n');

    const sources = Array.from(document.querySelectorAll('.source-item')).map(item => {
        const num = item.querySelector('.source-number')?.innerText || '';
        const link = item.querySelector('.source-link')?.href || '';
        const claim = item.querySelector('.source-claim')?.innerText || '';
        return `${num} ${link}${claim ? ` - ${claim}` : ''}`;
    }).join('\n');

    const fullText = `# ${title}\n\n${summary}\n\n${sections}\n\n### Sources\n\n${sources}`;

    navigator.clipboard.writeText(fullText).then(() => {
        const copyBtn = document.getElementById('copyBtn');
        if (copyBtn) {
            const originalText = copyBtn.innerText;
            copyBtn.innerText = 'Copied';
            setTimeout(() => { copyBtn.innerText = originalText; }, 2000);
        }
    });
}

function showError(userFriendlyMsg, technicalDetails) {
    const errorBanner = document.getElementById('errorBanner');
    const errorMessage = document.getElementById('errorMessage');
    const detailsTechnicalError = document.getElementById('detailsTechnicalError');
    const detailsErrorRow = document.getElementById('detailsErrorRow');

    if (errorMessage) errorMessage.innerText = userFriendlyMsg;
    if (errorBanner) errorBanner.style.display = 'flex';

    if (technicalDetails && detailsTechnicalError && detailsErrorRow) {
        detailsTechnicalError.innerText = technicalDetails;
        detailsErrorRow.style.display = 'block';
    }
}

async function fetchTraceQuietly() {
    if (!currentRunId) return;
    try {
        const resp = await fetch(getApiUrl(`/research/${currentRunId}/trace`), {
            headers: { 'X-API-Key': apiKey }
        });
        if (!resp.ok) return;

        const data = await resp.json();
        const latencyEl = document.getElementById('detailsLatency');
        if (latencyEl && data.total_latency_ms) {
            latencyEl.innerText = `${(data.total_latency_ms / 1000).toFixed(1)}s`;
        }

        const traceLog = document.getElementById('traceLog');
        if (traceLog && data.steps && data.steps.length) {
            traceLog.innerHTML = data.steps.map(step => `
                <div class="trace-entry">
                    <div class="trace-entry-header">
                        <strong>${escapeHtml(step.node_name)}</strong>
                        <span>${step.latency_ms.toFixed(0)}ms</span>
                    </div>
                    <div>${escapeHtml(step.output_summary || step.input_summary || '')}</div>
                </div>
            `).join('');
        }
    } catch (e) {
        // quiet ignore
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
