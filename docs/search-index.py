#!/usr/bin/env python3
"""Build the client-side search index from the rendered site.

Zola's own `build_search_index` covers markdown only, and it indexes whole pages.
Reading `public/` instead sees exactly what a reader sees, and it splits each
page at its `<h2>` boundaries so a result can link to the section rather than
the top of a long page.

Usage:  python3 search-index.py [public_dir]
Writes: <public_dir>/search-index.json
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

SPACE = re.compile(r"\s+")

# A section with less text than this is a stub, not something worth its own
# result row.
MIN_SECTION = 40


class Page(HTMLParser):
    """One linear pass: document title, breadcrumb group, and text per `<h2>`.

    Everything that is not prose is skipped as a subtree. That includes `svg`,
    whose label fragments would otherwise flood the index with single words,
    and both `nav` elements, one of which is read for the breadcrumb first.
    """

    SKIP = {"script", "style", "svg", "button", "noscript", "nav"}

    # Adjacent table cells and list items carry no whitespace between them, so
    # their text would otherwise concatenate into one unsearchable word.
    BLOCK = {
        "p", "div", "br", "li", "ul", "ol", "dl", "dt", "dd", "pre", "table",
        "tr", "td", "th", "thead", "tbody", "section", "article", "figure",
        "blockquote", "h1", "h2", "h3", "h4", "h5", "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.group = ""
        self.blocks: list[dict] = [{"id": "", "heading": "", "text": []}]
        self._in_title = False
        self._in_h2 = False
        self._main = 0
        self._skip = 0
        self._crumb_at = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)

        if self._main == 0:
            if tag == "title":
                self._in_title = True
            elif tag == "main" and attr.get("id") == "content":
                self._main = 1
            return

        if self._skip:
            if tag in self.SKIP:
                self._skip += 1
            return

        if tag in self.SKIP:
            self._skip = 1
            if tag == "nav" and "breadcrumb" in (attr.get("class") or ""):
                self._crumb_at = 1
            return

        if tag == "main":
            self._main += 1
        elif tag == "h2":
            self.blocks.append({"id": attr.get("id") or "", "heading": "", "text": []})
            self._in_h2 = True
        elif tag in self.BLOCK:
            self.blocks[-1]["text"].append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._main == 0:
            if tag == "title":
                self._in_title = False
            return

        if self._skip:
            if tag in self.SKIP:
                if self._crumb_at == self._skip:
                    self._crumb_at = 0
                self._skip -= 1
            return

        if tag == "main":
            self._main -= 1
        elif tag == "h2":
            self._in_h2 = False
        elif tag in self.BLOCK:
            self.blocks[-1]["text"].append(" ")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._crumb_at:
            self.group += data
        elif self._main and not self._skip:
            if self._in_h2:
                self.blocks[-1]["heading"] += data
            else:
                self.blocks[-1]["text"].append(data)


def page_records(html: str, url: str) -> list[dict]:
    page = Page()
    page.feed(html)
    if page._main == 0 and len(page.blocks) == 1 and not page.blocks[0]["text"]:
        return []

    # `<title>` is "<page> | CUCA"; the site name is noise in a result row.
    name = SPACE.sub(" ", page.title).strip().split(" | ")[0].strip() or url
    # The breadcrumb reads "CUCA / Group" or "CUCA / Group / Parent"; the group
    # is the one worth indexing.
    crumb = [part.strip() for part in page.group.split("/") if part.strip()]
    group = crumb[1] if len(crumb) > 1 else ""

    records = []
    for block in page.blocks:
        text = SPACE.sub(" ", "".join(block["text"])).strip()
        heading = SPACE.sub(" ", block["heading"]).strip()
        if len(text) < MIN_SECTION and not heading:
            continue
        records.append(
            {
                "u": url + (f"#{block['id']}" if block["id"] else ""),
                "t": name,
                "g": group,
                "s": heading,
                "b": text,
            }
        )
    return records


def build(public: Path) -> list[dict]:
    records: list[dict] = []
    for html_file in sorted(public.rglob("*.html")):
        if html_file.name == "404.html":
            continue
        rel = html_file.relative_to(public).parent.as_posix()
        url = "" if rel == "." else rel + "/"
        records.extend(page_records(html_file.read_text(encoding="utf-8"), url))
    return records


def main() -> int:
    public = Path(sys.argv[1] if len(sys.argv) > 1 else "public")
    if not public.is_dir():
        print(f"no such directory: {public}", file=sys.stderr)
        return 1

    records = build(public)
    out = public / "search-index.json"
    out.write_text(json.dumps(records, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"indexed {len(records)} sections into {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
