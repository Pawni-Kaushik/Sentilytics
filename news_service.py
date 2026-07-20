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
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"


def search_news(query: str, max_results: int = 10, lang: str = "en", country: str = "in"):
    """
    Search GNews for articles about a keyword/brand/topic.

    Args:
        query: search term, e.g. "Maruti Suzuki"
        max_results: how many articles to fetch (GNews free tier caps
            a single request at 10)
        lang: article language code
        country: country code to bias results toward (e.g. "in" for
            India-relevant coverage of a query like "Maruti Suzuki")

    Returns:
        [
            {
                "title": "...",
                "url": "...",
                "preview": "...",       # short description/snippet
                "content": "...",       # longer body snippet (may be
                                         # truncated by GNews on the free tier)
                "source": "...",
                "published_at": "...",
            },
            ...
        ]
    """

    if not GNEWS_API_KEY:
        print("\n========================")
        print("NEWS SEARCH FAILED")
        print("========================")
        print("GNEWS_API_KEY is not set. Add it to your .env file locally, "
              "or to Streamlit Cloud's Settings -> Secrets when deployed.")
        print("========================\n")
        return []

    params = {
        "q": query,
        "lang": lang,
        "country": country,
        "max": min(max_results, 10),
        "apikey": GNEWS_API_KEY,
    }

    try:
        response = requests.get(GNEWS_SEARCH_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

    except Exception as e:
        print("\n========================")
        print("NEWS SEARCH FAILED")
        print("========================")
        print(e)
        print("========================\n")
        return []

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

    print("\n========================")
    print("NEWS SEARCH")
    print("========================")
    print(f"Articles Found: {len(articles)}")
    print("========================")

    for item in articles:
        print(item)

    print()

    return articles


if __name__ == "__main__":
    # Quick manual test
    results = search_news("Maruti Suzuki", max_results=5)
    for r in results:
        print("\n" + "=" * 60)
        print(r["title"])
        print(r["url"])
        print(r["preview"][:200])
