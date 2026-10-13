import logging

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


def extract_full_page(
    url: str,
    timeout_seconds: float = 8.0,
    token_cap: int = 2500,
) -> tuple[str, ToolStatus]:
    """
    Fetches full page HTML and extracts readable article text via trafilatura.
    Truncates extracted content to token cap before returning to model.
    Returns: (extracted_text, status)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)

            if resp.status_code in (401, 402):
                return "", ToolStatus.PAYWALLED
            elif resp.status_code in (403, 429):
                return "", ToolStatus.BLOCKED if resp.status_code == 403 else ToolStatus.RATE_LIMITED
            elif resp.status_code >= 400:
                return "", ToolStatus.PARSE_ERROR

            html_content = resp.text

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
        # Fallback to simple extraction
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
