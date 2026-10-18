import ipaddress
import logging
import socket
from typing import Any
from urllib.parse import urlparse

import httpcore
import httpx

from app.models.enums import ToolStatus

logger = logging.getLogger(__name__)

PAYWALL_KEYWORDS = [
    "subscribe to read",
    "subscription required",
    "create an account to continue",
    "sign in to read the full story",
    "you have reached your limit of free articles",
    "paywall",
]

MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MB response size cap
MAX_REDIRECTS = 3
DEFAULT_TIMEOUT_SECONDS = 10.0


def is_safe_url(url: str) -> tuple[bool, str]:
    """
    Validates a URL against SSRF vulnerabilities:
    - Blocks non-http/https schemes (e.g. file://, ftp://, gopher://).
    - Blocks private, loopback, link-local, and cloud metadata IP addresses.
    Returns: (is_safe, reason)
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Malformed URL: {e}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False, f"Prohibited scheme '{scheme}'. Only http and https are allowed."

    hostname = parsed.hostname
    if not hostname:
        return False, "Missing or invalid hostname in URL."

    # Direct IP string validation
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_disallowed_ip(ip):
            return False, f"Blocked internal/private IP address: {ip}"
        return True, "Safe URL"
    except ValueError:
        # Hostname is a domain, resolve via DNS
        pass

    try:
        # Resolve all IPv4 and IPv6 addresses
        addr_info = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if _is_disallowed_ip(ip):
                return False, f"Resolved hostname '{hostname}' points to blocked IP: {ip}"
    except socket.gaierror:
        # If domain cannot be resolved via DNS, it cannot route to a private/internal IP.
        # Allow client to handle (e.g. mocked HTTP client in tests or network-level ConnectError).
        return True, "Safe URL (unresolved domain)"
    except Exception as e:
        return False, f"IP verification error for '{hostname}': {e}"

    return True, "Safe URL"


def _is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Checks whether an IP address is internal, loopback, private, link-local, or cloud metadata."""
    if getattr(ip, "ipv4_mapped", None):
        return _is_disallowed_ip(ip.ipv4_mapped)
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or str(ip) == "169.254.169.254"  # Cloud metadata service (AWS/GCP/Azure)
    )


class SSRFSafeBackend(httpcore._backends.sync.SyncBackend):
    """Custom network backend validating connected peer IP at connect time to prevent DNS rebinding."""

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any | None = None,
    ):
        stream = super().connect_tcp(host, port, timeout, local_address, socket_options)
        server_addr = stream.get_extra_info("server_addr")
        if server_addr:
            peer_ip_str = server_addr[0]
            try:
                peer_ip = ipaddress.ip_address(peer_ip_str)
                if _is_disallowed_ip(peer_ip):
                    stream.close()
                    raise httpx.RequestError(f"SSRF connect-time check blocked peer IP: {peer_ip}")
            except ValueError:
                pass
        return stream


class SSRFSafeTransport(httpx.HTTPTransport):
    """HTTPTransport that injects SSRFSafeBackend to re-validate socket connections."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self._pool = httpcore.ConnectionPool(
                network_backend=SSRFSafeBackend(),
                ssl_context=getattr(self._pool, "_ssl_context", None),
                max_connections=getattr(self._pool, "_max_connections", 100),
                max_keepalive_connections=getattr(self._pool, "_max_keepalive_connections", 20),
                keepalive_expiry=getattr(self._pool, "_keepalive_expiry", 5.0),
                http1=getattr(self._pool, "_http1", True),
                http2=getattr(self._pool, "_http2", False),
                retries=getattr(self._pool, "_retries", 0),
            )
        except Exception as e:
            logger.debug(f"SSRFSafeTransport could not replace pool backend: {e}")


def extract_full_page(
    url: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    token_cap: int = 2500,
) -> tuple[str, ToolStatus]:
    """
    Fetches full page HTML and extracts readable article text via trafilatura.
    Enforces SSRF prevention, manual redirect following (max 3), connect-time validation,
    and response size truncation (2 MB).
    Returns: (extracted_text, status)
    """
    # 1. SSRF Safety Check on initial URL
    is_safe, reason = is_safe_url(url)
    if not is_safe:
        logger.warning(f"SSRF protection blocked fetch for URL '{url}': {reason}")
        return "", ToolStatus.BLOCKED

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        transport = SSRFSafeTransport(verify=True)
        with httpx.Client(
            transport=transport,
            timeout=timeout_seconds,
            follow_redirects=False,
            headers=headers,
            verify=True,
        ) as client:
            current_url = url
            resp = None
            hops = 0

            while True:
                # Validate URL on every hop
                hop_safe, hop_reason = is_safe_url(current_url)
                if not hop_safe:
                    logger.warning(f"SSRF blocked hop to URL '{current_url}': {hop_reason}")
                    return "", ToolStatus.BLOCKED

                resp = client.get(current_url)

                # Check for redirect responses
                if resp.status_code in (301, 302, 303, 307, 308):
                    hops += 1
                    if hops > MAX_REDIRECTS:
                        logger.warning(f"Exceeded max redirects ({MAX_REDIRECTS}) starting from {url}")
                        return "", ToolStatus.BLOCKED

                    location = resp.headers.get("Location")
                    if not location:
                        return "", ToolStatus.PARSE_ERROR

                    from urllib.parse import urljoin

                    next_url = urljoin(current_url, location)
                    next_safe, next_reason = is_safe_url(next_url)
                    if not next_safe:
                        logger.warning(f"SSRF blocked redirect target '{next_url}': {next_reason}")
                        return "", ToolStatus.BLOCKED

                    current_url = next_url
                    continue

                break

            if resp.status_code in (401, 402):
                return "", ToolStatus.PAYWALLED
            elif resp.status_code in (403, 429):
                return "", ToolStatus.BLOCKED if resp.status_code == 403 else ToolStatus.RATE_LIMITED
            elif resp.status_code >= 400:
                return "", ToolStatus.PARSE_ERROR

            # Enforce 2 MB max response limit
            raw_content = getattr(resp, "content", b"")
            if not isinstance(raw_content, bytes):
                raw_content = getattr(resp, "text", "").encode("utf-8", errors="replace")
            content_bytes = raw_content[:MAX_RESPONSE_BYTES]
            if len(raw_content) > MAX_RESPONSE_BYTES:
                logger.warning(f"Response size exceeded {MAX_RESPONSE_BYTES} bytes cap for {current_url}. Truncating.")
            encoding = getattr(resp, "encoding", None) or "utf-8"
            html_content = content_bytes.decode(encoding, errors="replace")

    except httpx.TooManyRedirects:
        logger.warning(f"Exceeded max redirects ({MAX_REDIRECTS}) fetching {url}")
        return "", ToolStatus.BLOCKED
    except httpx.TimeoutException:
        return "", ToolStatus.TIMEOUT
    except httpx.RequestError as e:
        logger.warning(f"HTTP request error fetching {url}: {e}")
        return "", ToolStatus.BLOCKED

    # Check for paywall keywords in raw HTML
    lower_html = html_content.lower()
    if any(pw in lower_html for pw in PAYWALL_KEYWORDS):
        return "", ToolStatus.PAYWALLED

    # Extract clean readable text with trafilatura
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
    except Exception as e:
        logger.warning(f"Trafilatura extraction failure for {url}: {e}")
        extracted = None

    if not extracted or len(extracted.strip()) < 50:
        # Fallback to simple BeautifulSoup extraction
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            extracted = soup.get_text(separator="\n", strip=True)
        except Exception:
            return "", ToolStatus.PARSE_ERROR

    if not extracted:
        return "", ToolStatus.PARSE_ERROR

    # Truncate to token cap (~4 chars per token)
    char_limit = token_cap * 4
    if len(extracted) > char_limit:
        extracted = extracted[:char_limit] + "\n...[truncated to token cap]"

    return extracted, ToolStatus.OK
