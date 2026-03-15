"""Crawl MQL5 articles listing pages and build a master catalogue.

Usage:
    uv run python scripts/crawl_mql5_articles.py [--pages 69] [--delay 1.5]

Output:
    Research/mql5_catalogue.json — master list of all articles
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.mql5.com/en/articles"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
OUTPUT_PATH = Path(__file__).parent.parent / "Research" / "mql5_catalogue.json"


def fetch_page(page_num: int, delay: float) -> str | None:
    """Fetch a single article listing page."""
    if page_num == 1:
        url = BASE_URL
    else:
        url = f"{BASE_URL}/page{page_num}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        time.sleep(delay)  # Be polite
        return resp.text
    except Exception as e:
        print(f"  ERROR fetching page {page_num}: {e}")
        return None


def parse_articles(html: str) -> list[dict]:
    """Parse article entries from a listing page.

    MQL5 article listings use <section> blocks with:
    - <header><h3><a href="/en/articles/NNNNN">Title</a></h3></header>
    - Description text as sibling content within the section
    """
    soup = BeautifulSoup(html, "html.parser")
    articles = []

    for section in soup.find_all("section"):
        header = section.find("header")
        if not header:
            continue
        link = header.find("a", href=re.compile(r"^/en/articles/\d+$"))
        if not link:
            continue

        href = link["href"]
        article_id = re.match(r"^/en/articles/(\d+)$", href).group(1)
        title = link.get_text(strip=True)
        if not title or len(title) < 5:
            img = link.find("img")
            if img and img.get("alt"):
                title = img["alt"]
            else:
                continue

        # Description is text content of the section outside the header
        desc_parts = []
        for child in section.children:
            if child == header:
                continue
            text = child.get_text(strip=True) if hasattr(child, "get_text") else ""
            if text and len(text) > 20:
                desc_parts.append(text)
        description = " ".join(desc_parts)[:500]

        articles.append({
            "id": article_id,
            "url": f"https://www.mql5.com{href}",
            "title": title,
            "description": description,
            "category": "",
            "status": "new",
        })

    return articles


def deduplicate(articles: list[dict]) -> list[dict]:
    """Remove duplicate articles by ID."""
    seen = set()
    unique = []
    for a in articles:
        if a["id"] not in seen:
            seen.add(a["id"])
            unique.append(a)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Crawl MQL5 article listings")
    parser.add_argument("--pages", type=int, default=69, help="Number of pages to crawl")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between requests (seconds)")
    parser.add_argument("--start", type=int, default=1, help="Starting page number")
    args = parser.parse_args()

    all_articles: list[dict] = []

    # Load existing catalogue if resuming
    if OUTPUT_PATH.exists() and args.start > 1:
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
            all_articles = existing.get("articles", [])
            print(f"Loaded {len(all_articles)} existing articles")

    for page in range(args.start, args.pages + 1):
        print(f"Fetching page {page}/{args.pages}...", end=" ", flush=True)
        html = fetch_page(page, args.delay)
        if html is None:
            continue

        articles = parse_articles(html)
        print(f"found {len(articles)} articles")
        all_articles.extend(articles)

        # Save checkpoint every 5 pages
        if page % 5 == 0:
            unique = deduplicate(all_articles)
            catalogue = {
                "source": "https://www.mql5.com/en/articles",
                "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pages_crawled": page,
                "total_articles": len(unique),
                "articles": unique,
            }
            with open(OUTPUT_PATH, "w") as f:
                json.dump(catalogue, f, indent=2)
            print(f"  Checkpoint: {len(unique)} unique articles saved")

    # Final save
    unique = deduplicate(all_articles)
    catalogue = {
        "source": "https://www.mql5.com/en/articles",
        "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pages_crawled": args.pages,
        "total_articles": len(unique),
        "articles": unique,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(catalogue, f, indent=2)

    print(f"\nDone! {len(unique)} unique articles saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
