import socket

import httpx

from app.tools.fetch_extract import (
    MAX_RESPONSE_BYTES,
    ToolStatus,
    extract_full_page,
    is_safe_url,
)


def test_ssrf_protection_rules():
    """Validates SSRF protection blocking private, loopback, link-local, cloud metadata IPs, and non-http schemes."""
    # 1. Non-http schemes
    safe, reason = is_safe_url("file:///etc/passwd")
    assert safe is False
    assert "scheme" in reason.lower()

    safe, reason = is_safe_url("ftp://example.com/file")
    assert safe is False

    safe, reason = is_safe_url("gopher://example.com:70/1")
    assert safe is False

    # 2. Loopback and internal IPs
    safe, reason = is_safe_url("http://127.0.0.1/admin")
    assert safe is False
    assert "blocked" in reason.lower() or "private" in reason.lower()

    safe, reason = is_safe_url("http://localhost/secret")
    assert safe is False

    # 3. Cloud metadata service
    safe, reason = is_safe_url("http://169.254.169.254/latest/meta-data/")
    assert safe is False

    # 4. RFC 1918 Private ranges
    safe, reason = is_safe_url("http://10.0.0.1/status")
    assert safe is False

    safe, reason = is_safe_url("http://192.168.1.1/router")
    assert safe is False

    safe, reason = is_safe_url("http://172.16.0.1/internal")
    assert safe is False

    # 5. Public URL allowed
    safe, reason = is_safe_url("https://example.com/docs")
    assert safe is True


def test_ssrf_redirect_to_loopback_and_metadata(monkeypatch):
    """Validates that extract_full_page blocks redirects to 127.0.0.1 and 169.254.169.254."""
    # 1. Redirect to 127.0.0.1
    resp_redirect_loopback = httpx.Response(
        status_code=302,
        headers={"Location": "http://127.0.0.1/admin"},
        request=httpx.Request("GET", "https://public-source.org/login"),
    )
    monkeypatch.setattr("httpx.Client.get", lambda *args, **kwargs: resp_redirect_loopback)
    text, status = extract_full_page("https://public-source.org/login")
    assert status == ToolStatus.BLOCKED
    assert text == ""

    # 2. Redirect to 169.254.169.254
    resp_redirect_metadata = httpx.Response(
        status_code=302,
        headers={"Location": "http://169.254.169.254/latest/meta-data/"},
        request=httpx.Request("GET", "https://public-source.org/api"),
    )
    monkeypatch.setattr("httpx.Client.get", lambda *args, **kwargs: resp_redirect_metadata)
    text, status = extract_full_page("https://public-source.org/api")
    assert status == ToolStatus.BLOCKED
    assert text == ""


def test_ssrf_ip_resolution_and_mapped_addresses(monkeypatch):
    """Validates SSRF prevention on private DNS resolution, IPv6 loopback, and IPv4-mapped IPv6."""
    # 1. Hostname resolving to private IP
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, proto=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.4.2", 80))],
    )
    safe, reason = is_safe_url("https://deceptive-internal-dns.org/report")
    assert safe is False
    assert "blocked" in reason.lower()

    # 2. IPv6 loopback ::1
    safe, reason = is_safe_url("http://[::1]/debug")
    assert safe is False
    assert "blocked" in reason.lower()

    # 3. IPv4-mapped IPv6 ::ffff:127.0.0.1
    safe, reason = is_safe_url("http://[::ffff:127.0.0.1]/secret")
    assert safe is False
    assert "blocked" in reason.lower()


def test_extract_full_page_truncates_oversized_response(monkeypatch):
    """Validates that extract_full_page handles responses > 2 MB by truncating without failure."""
    large_body = b"<html><body><p>" + (b"A" * (3 * 1024 * 1024)) + b"</p></body></html>"
    mock_resp = httpx.Response(
        status_code=200,
        content=large_body,
        request=httpx.Request("GET", "https://public-benchmark.org/large"),
    )
    monkeypatch.setattr("httpx.Client.get", lambda *args, **kwargs: mock_resp)

    text, status = extract_full_page("https://public-benchmark.org/large")
    assert status == ToolStatus.OK
    assert len(text) > 0
    assert len(text) <= MAX_RESPONSE_BYTES
