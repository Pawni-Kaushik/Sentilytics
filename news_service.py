"""
news_service.py
----------------
Fetches live news articles about any keyword/brand/topic using the
GNews API (https://gnews.io).

Replaces the old reddit_service.py + reddit_scraper.py combo:
  - Old flow: Google Search (SerpAPI) finds Reddit URLs -> Playwright
    headless browser scrapes each URL's HTML for text.
    PROBLEM: Reddit blocks datacenter/cloud IPs (Streamlit Community
    Cloud included), so the scrape step failed 100% of the time once
    deployed, even though it worked locally.

  - New flow: one authenticated HTTPS request to GNews's REST API.
    No scraping, no browser, no IP blocking -- GNews returns
    structured JSON (title, description, content, url, source,
    publishedAt) directly. This is a real, deployable API call and
    works identically on Streamlit Community Cloud as it does locally.

Get a free API key (100 requests/day, deployable -- not localhost-only)
at https://gnews.io. Add it to .env locally as:

    GNEWS_API_KEY=your_key_here

...and to Streamlit Cloud's Settings -> Secrets when deployed:

    GNEWS_API_KEY = "your_key_here"

IMPORTANT -- the free plan's biggest real-world limitation is NOT "the
key is wrong", it's the 100-requests/day cap. Once that's used up,
GNews returns an error for EVERY request until it resets at 00:00 UTC.
If you were testing earlier in the day and some keywords "still show
results" while brand-new ones fail, that's almost always this: the
working ones are old results still sitting in session_state (see
pages/6_Live_News_Search.py), not a fresh call. This module now
reports exactly which of these happened instead of collapsing them
into one generic message:
  - "missing_key"    -> GNEWS_API_KEY isn't set anywhere Streamlit can see
  - "invalid_key"    -> key is set but GNews rejected it (401)
  - "quota_exceeded" -> daily 100-request cap hit (403 / 429)
  - "network_error"  -> request timed out / couldn't reach GNews
  - None (success)   -> call worked; articles list may still be empty
                        if GNews simply has no coverage for that query
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"

# GNews's free plan allows 100 requests/day, resetting at 00:00 UTC.
# Bump this if you upgrade plans.
DAILY_REQUEST_LIMIT = 100

# GNews's API does not return any "requests remaining" header, so there
# is no way to ask GNews directly. Instead this app counts its own
# calls locally in a small JSON file, reset whenever the UTC date
# changes. Caveat: this lives inside the app's own container, so it
# resets to 0 if the app restarts, redeploys, or wakes from sleep --
# treat it as a close estimate, not the authoritative number. The
# exact count is always visible at https://gnews.io/dashboard.
_USAGE_FILE = Path(__file__).parent / "gnews_usage.json"


def _today_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_usage():
    data = {}
    if _USAGE_FILE.exists():
        try:
            data = json.loads(_USAGE_FILE.read_text())
        except Exception:
            data = {}
    if data.get("date") != _today_utc_str():
        data = {"date": _today_utc_str(), "count": 0}
    return data


def _record_request():
    """Called once per actual call made to GNews (any response received)."""
    data = _load_usage()
    data["count"] = data.get("count", 0) + 1
    try:
        _USAGE_FILE.write_text(json.dumps(data))
    except Exception as e:
        print(f"Could not persist GNews usage counter: {e}")


def get_usage_today():
    """
    Returns (used, remaining, limit) for today (UTC), based on this
    app's own local counter of calls it has made -- see the caveat in
    the module docstring above about container restarts resetting this.
    """
    data = _load_usage()
    used = data.get("count", 0)
    remaining = max(DAILY_REQUEST_LIMIT - used, 0)
    return used, remaining, DAILY_REQUEST_LIMIT


def _get_api_key():
    """
    Reads the key from a plain environment variable first (covers local
    .env via python-dotenv, and Streamlit Cloud's root-level secrets,
    which Streamlit also injects into os.environ). Falls back to
    st.secrets directly in case it was set under a nested section, or
    the environment injection didn't happen for some reason.
    """
    key = os.getenv("GNEWS_API_KEY")
    if key:
        return key.strip()

    try:
        import streamlit as st
        if "GNEWS_API_KEY" in st.secrets:
            return str(st.secrets["GNEWS_API_KEY"]).strip()
    except Exception:
        pass

    return None


def search_news(query: str, max_results: int = 10, lang: str = "en", country: str = "in"):
    """
    Search GNews for articles about a keyword/brand/topic.

    Returns:
        (articles, error_code)

        articles: list of dicts (title, url, preview, content, source,
            published_at) -- possibly empty even on success, if GNews
            has no matching coverage.
        error_code: None on a successful API call (regardless of how
            many articles came back), otherwise one of:
            "missing_key", "invalid_key", "quota_exceeded",
            "network_error"
    """

    api_key = _get_api_key()
    if not api_key:
        print("NEWS SEARCH FAILED: GNEWS_API_KEY is not set (checked os.environ and st.secrets).")
        return [], "missing_key"

    params = {
        "q": query,
        "lang": lang,
        "country": country,
        "max": min(max_results, 10),
        "apikey": api_key,
    }

    try:
        response = requests.get(GNEWS_SEARCH_URL, params=params, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"NEWS SEARCH FAILED (network error): {e}")
        return [], "network_error"

    # A response was actually received from GNews, so this counts
    # against the daily quota regardless of status code.
    _record_request()

    if response.status_code == 401:
        print(f"NEWS SEARCH FAILED (401 invalid key): {response.text[:300]}")
        return [], "invalid_key"

    if response.status_code in (403, 429):
        print(f"NEWS SEARCH FAILED ({response.status_code} quota/rate limit): {response.text[:300]}")
        return [], "quota_exceeded"

    if not response.ok:
        print(f"NEWS SEARCH FAILED ({response.status_code}): {response.text[:300]}")
        return [], "network_error"

    try:
        data = response.json()
    except ValueError as e:
        print(f"NEWS SEARCH FAILED (bad JSON): {e}")
        return [], "network_error"

    articles = []
    for item in data.get("articles", []):
        articles.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "preview": item.get("description", "") or "",
            "content": item.get("content", "") or "",
            "source": (item.get("source") or {}).get("name", ""),
            "published_at": item.get("publishedAt", ""),
        })

    print(f"NEWS SEARCH OK -- {len(articles)} article(s) for '{query}'")
    return articles, None


if __name__ == "__main__":
    # Quick manual test
    results, err = search_news("Maruti Suzuki", max_results=5)
    if err:
        print("ERROR:", err)
    for r in results:
        print("\n" + "=" * 60)
        print(r["title"])
        print(r["url"])
        print(r["preview"][:200])
