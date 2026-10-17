import re
from typing import Optional, Tuple
import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def extract_page_content(
    url: str,
    timeout_seconds: float = 8.0,
    max_length: int = 4000
) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetches and extracts clean, readable text from a target URL.
    Returns: (extracted_text, error_reason)
    Never raises an uncaught exception.
    """
    try:
        with httpx.Client(
            headers=HEADERS,
            follow_redirects=True,
            timeout=timeout_seconds,
            verify=True
        ) as client:
            resp = client.get(url)

            if resp.status_code == 403 or resp.status_code == 401:
                return None, f"paywalled_or_forbidden (HTTP {resp.status_code})"
            if resp.status_code == 429:
                return None, "rate_limited"
            if resp.status_code >= 400:
                return None, f"http_error_{resp.status_code}"

            html = resp.text
            soup = BeautifulSoup(html, "html.parser")

            # Remove non-content elements
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "aside"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                return None, "empty_content"

            # Truncate to reasonable length for downstream LLM processing
            clean_text = text[:max_length]
            return clean_text, None

    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.RequestError as e:
        return None, f"network_error: {type(e).__name__}"
    except Exception as e:
        return None, f"extraction_error: {str(e)}"
