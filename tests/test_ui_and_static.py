import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dashboard_ui_html_structure():
    """Validates that GET / delivers the rich Tailwind dashboard HTML with all required interactive element IDs."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        html = resp.text

        # Verify key interactive elements exist
        assert 'id="topicInput"' in html
        assert 'id="runBtn"' in html
        assert 'id="progressBox"' in html
        assert 'id="progressBar"' in html
        assert 'id="resultsCard"' in html
        assert 'id="reportPane"' in html
        assert 'id="tracePane"' in html
        assert 'id="node-planner"' in html
        assert 'id="node-researcher"' in html
        assert 'id="node-supervisor"' in html
        assert 'id="node-writer"' in html

        # Verify Tailwind and font imports
        assert "cdn.tailwindcss.com" in html
        assert "JetBrains+Mono" in html
        assert "Outfit" in html


@pytest.mark.unit
@pytest.mark.asyncio
async def test_static_assets_delivery():
    """Validates delivery of static CSS and JS files."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Check styles.css
        css_resp = await client.get("/static/css/styles.css")
        assert css_resp.status_code == 200
        assert "text/css" in css_resp.headers["content-type"]
        assert "--bg-primary" in css_resp.text

        # Check app.js
        js_resp = await client.get("/static/js/app.js")
        assert js_resp.status_code == 200
        assert (
            "application/javascript" in js_resp.headers["content-type"]
            or "text/javascript" in js_resp.headers["content-type"]
        )
        assert "startResearch" in js_resp.text
