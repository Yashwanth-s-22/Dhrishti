"""
Drishti MCP Tool: GDELT News Search
===================================
Queries the GDELT DOC 2.0 API to retrieve real-time and recent global news articles
related to agricultural trade, geopolitical shocks, and commodity policies.
Handles network timeouts, HTTP 429 rate limits, non-200 responses, and empty results
gracefully without fabricating fake news.
"""

import requests
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    GDELT_BASE_URL,
    GDELT_TIMEOUT_SECONDS,
    GDELT_DEFAULT_MAX_RECORDS,
    GDELT_DEFAULT_TIMESPAN,
)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _clean_query_for_gdelt(raw_query: str) -> str:
    """Simplify query to essential search keywords for GDELT API."""
    # Remove punctuation and special characters
    cleaned = re.sub(r"[^\w\s]", " ", raw_query)
    # Collapse multiple spaces
    cleaned = " ".join(cleaned.split())
    # Keep up to top 6 keywords to prevent over-constraining GDELT
    words = cleaned.split()
    if len(words) > 6:
        cleaned = " ".join(words[:6])
    return cleaned


def search_gdelt_news(
    query: str,
    timespan: str = GDELT_DEFAULT_TIMESPAN,
    max_records: int = GDELT_DEFAULT_MAX_RECORDS,
    timeout: int = GDELT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Search GDELT DOC 2.0 API for recent news articles.

    Args:
        query: Search keywords (e.g. "India rice export ban", "Ukraine wheat shipment")
        timespan: Time window (e.g. "24h", "7d", "1m", "3m")
        max_records: Maximum number of articles to return (max 250)
        timeout: Request timeout in seconds

    Returns:
        Structured dictionary containing query status, article list, error details, and provenance.
    """
    if not query or not query.strip():
        return {
            "status": "error",
            "message": "Search query cannot be empty.",
            "article_count": 0,
            "articles": [],
            "provenance": "[GDELT DATA]",
        }

    clean_q = _clean_query_for_gdelt(query)

    params = {
        "query": clean_q,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(max(1, max_records), 250),
        "timespan": timespan.strip(),
        "sort": "DateDesc",
    }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
    }

    try:
        response = requests.get(GDELT_BASE_URL, params=params, headers=headers, timeout=timeout)

        if response.status_code == 200:
            try:
                data = response.json()
                raw_articles = data.get("articles", [])
                
                cleaned_articles: List[Dict[str, str]] = []
                for art in raw_articles:
                    cleaned_articles.append({
                        "title": str(art.get("title", "")).strip(),
                        "url": str(art.get("url", "")).strip(),
                        "source_country": str(art.get("sourcecountry", "Unknown")).strip(),
                        "seen_date": str(art.get("seendate", "")).strip(),
                        "domain": str(art.get("domain", "")).strip(),
                        "language": str(art.get("language", "")).strip(),
                    })

                return {
                    "status": "success" if cleaned_articles else "no_articles",
                    "query": clean_q,
                    "timespan": timespan,
                    "article_count": len(cleaned_articles),
                    "articles": cleaned_articles,
                    "error": None if cleaned_articles else "No articles matched the search query within the specified timespan.",
                    "provenance": "[GDELT DATA]",
                }
            except Exception as parse_err:
                return {
                    "status": "unavailable",
                    "query": clean_q,
                    "error": f"Failed to parse GDELT JSON response ({parse_err}).",
                    "article_count": 0,
                    "articles": [],
                    "provenance": "[GDELT DATA]",
                }
        elif response.status_code == 429:
            return {
                "status": "rate_limited",
                "query": clean_q,
                "error": "GDELT API rate limit reached (max 1 request per 5 seconds).",
                "article_count": 0,
                "articles": [],
                "provenance": "[GDELT DATA]",
            }
        else:
            return {
                "status": "unavailable",
                "query": clean_q,
                "error": f"GDELT API returned HTTP {response.status_code}.",
                "article_count": 0,
                "articles": [],
                "provenance": "[GDELT DATA]",
            }

    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "query": clean_q,
            "error": f"GDELT API request timed out after {timeout} seconds.",
            "article_count": 0,
            "articles": [],
            "provenance": "[GDELT DATA]",
        }
    except requests.exceptions.RequestException as req_err:
        return {
            "status": "network_error",
            "query": clean_q,
            "error": f"GDELT connection error: {req_err}",
            "article_count": 0,
            "articles": [],
            "provenance": "[GDELT DATA]",
        }
