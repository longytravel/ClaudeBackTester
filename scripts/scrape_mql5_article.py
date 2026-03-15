"""Scrape a single MQL5 article by ID and save as markdown.

Usage:
    uv run python scripts/scrape_mql5_article.py --id 21391
    uv run python scripts/scrape_mql5_article.py --id 21391 --force  # overwrite existing
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

CATALOGUE_PATH = Path("Research/mql5_catalogue.json")
OUTPUT_DIR = Path("Research/articles")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def load_catalogue() -> dict:
    with open(CATALOGUE_PATH, encoding="utf-8") as f:
        return json.load(f)


def find_article(catalogue: dict, article_id: str) -> dict | None:
    for a in catalogue["articles"]:
        if str(a["id"]) == str(article_id):
            return a
    return None


def scrape_article(url: str) -> tuple[str, str]:
    """Fetch article page, return (title, markdown_content)."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract title
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else "Untitled"

    # Find article body - MQL5 uses different class names
    body = (
        soup.find("div", class_="mqArticle__body")
        or soup.find("div", class_="article-body")
        or soup.find("div", id="articleContent")
        or soup.find("article")
    )

    if not body:
        # Fallback: grab the largest div with paragraphs
        divs = soup.find_all("div")
        best = max(divs, key=lambda d: len(d.find_all("p")), default=None)
        body = best

    if not body:
        return title, "*Could not extract article body.*"

    md = _html_to_markdown(body)
    return title, md


def _html_to_markdown(element) -> str:
    """Convert HTML element tree to simple markdown."""
    parts: list[str] = []

    for child in element.children:
        if isinstance(child, str):
            text = child.strip()
            if text:
                parts.append(text)
            continue

        tag = getattr(child, "name", None)
        if tag is None:
            continue

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            text = child.get_text(strip=True)
            if text:
                parts.append(f"\n{'#' * level} {text}\n")

        elif tag == "p":
            text = _inline_text(child)
            if text.strip():
                parts.append(f"\n{text}\n")

        elif tag == "pre" or (tag == "div" and "code" in child.get("class", [])):
            code = child.get_text()
            # Detect MQL5 code
            lang = "mql5" if any(kw in code for kw in ["OnTick", "MqlRates", "iMA", "OrderSend"]) else ""
            parts.append(f"\n```{lang}\n{code.strip()}\n```\n")

        elif tag == "code":
            parts.append(f"`{child.get_text()}`")

        elif tag == "ul":
            for li in child.find_all("li", recursive=False):
                text = _inline_text(li)
                parts.append(f"- {text}")
            parts.append("")

        elif tag == "ol":
            for i, li in enumerate(child.find_all("li", recursive=False), 1):
                text = _inline_text(li)
                parts.append(f"{i}. {text}")
            parts.append("")

        elif tag == "table":
            parts.append(_table_to_md(child))

        elif tag == "img":
            alt = child.get("alt", "image")
            src = child.get("src", "")
            parts.append(f"![{alt}]({src})")

        elif tag == "blockquote":
            text = child.get_text(strip=True)
            parts.append(f"\n> {text}\n")

        elif tag in ("div", "section", "span", "figure", "figcaption"):
            # Recurse into container elements
            inner = _html_to_markdown(child)
            if inner.strip():
                parts.append(inner)

        elif tag == "br":
            parts.append("\n")

        elif tag == "hr":
            parts.append("\n---\n")

        elif tag == "a":
            text = child.get_text(strip=True)
            href = child.get("href", "")
            if text and href:
                parts.append(f"[{text}]({href})")
            elif text:
                parts.append(text)

        elif tag in ("strong", "b"):
            text = child.get_text(strip=True)
            if text:
                parts.append(f"**{text}**")

        elif tag in ("em", "i"):
            text = child.get_text(strip=True)
            if text:
                parts.append(f"*{text}*")

    return "\n".join(parts)


def _inline_text(element) -> str:
    """Extract inline text with basic formatting."""
    parts: list[str] = []
    for child in element.children:
        if isinstance(child, str):
            parts.append(child)
        elif child.name == "strong" or child.name == "b":
            parts.append(f"**{child.get_text()}**")
        elif child.name == "em" or child.name == "i":
            parts.append(f"*{child.get_text()}*")
        elif child.name == "code":
            parts.append(f"`{child.get_text()}`")
        elif child.name == "a":
            text = child.get_text()
            href = child.get("href", "")
            parts.append(f"[{text}]({href})" if href else text)
        elif child.name == "br":
            parts.append("\n")
        else:
            parts.append(child.get_text())
    return "".join(parts).strip()


def _table_to_md(table) -> str:
    """Convert HTML table to markdown table."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    # Normalize column count
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    lines: list[str] = []
    # Header
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    # Body
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n" + "\n".join(lines) + "\n"


def update_catalogue_status(article_id: str, status: str) -> None:
    """Update an article's status in the catalogue."""
    cat = load_catalogue()
    for a in cat["articles"]:
        if str(a["id"]) == str(article_id):
            a["status"] = status
            break
    with open(CATALOGUE_PATH, "w", encoding="utf-8") as f:
        json.dump(cat, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape an MQL5 article to markdown")
    parser.add_argument("--id", required=True, help="Article ID from catalogue")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = parser.parse_args()

    article_id = args.id
    output_path = OUTPUT_DIR / f"{article_id}.md"

    if output_path.exists() and not args.force:
        print(f"Already scraped: {output_path}")
        print(f"Use --force to overwrite")
        sys.exit(0)

    # Look up in catalogue
    cat = load_catalogue()
    article = find_article(cat, article_id)
    if not article:
        print(f"Article {article_id} not found in catalogue")
        sys.exit(1)

    url = article["url"]
    print(f"Scraping: {article['title']}")
    print(f"URL: {url}")

    try:
        title, content = scrape_article(url)
    except requests.RequestException as e:
        print(f"Failed to fetch article: {e}")
        print("Fallback: paste the article content manually into Research/articles/{article_id}.md")
        sys.exit(1)

    # Build output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = f"# {title}\n\n"
    header += f"**Source**: [{url}]({url})\n"
    header += f"**Article ID**: {article_id}\n"
    header += f"**Category**: {article.get('category', 'unknown')}\n"
    header += f"**Scraped**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"

    output_path.write_text(header + content, encoding="utf-8")
    print(f"Saved: {output_path} ({len(content):,} chars)")

    # Update status
    update_catalogue_status(article_id, "scraped")
    print(f"Catalogue status updated to 'scraped'")


if __name__ == "__main__":
    main()
