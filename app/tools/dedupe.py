import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

STRIP_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "fbclid",
    "gclid",
    "source",
}


def canonicalize_url(url: str) -> str:
    """Normalizes URL by stripping tracking parameters, lowercasing domain, and removing trailing slash."""
    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]

        # Filter query params
        query_pairs = parse_qsl(parsed.query)
        cleaned_pairs = [(k, v) for k, v in query_pairs if k.lower() not in STRIP_PARAMS]
        query = urlencode(sorted(cleaned_pairs))

        path = parsed.path.lower().rstrip("/")
        if not path:
            path = "/"

        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return url.strip().lower()


def hash_content(text: str) -> str:
    """Generates a SHA-256 hash of normalized text content for deduplication."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
