"""Read the pages behind search results, not just their snippets.

A search snippet is a sentence or two picked by the search engine; the figures,
mechanisms and caveats a researcher needs are usually on the page itself. For the
top results a researcher reads, this fetches the page, extracts its readable text
(standard library only: no browser, no extra dependency), and keeps the passages
most relevant to the researcher's question, capped so a page cannot crowd out the
others. Anything that fails — a block, a timeout, a PDF, a script-only page —
returns None and the researcher falls back to the snippet.

``fetch_text`` is the one network boundary; recorded runs (maestro/cassette.py)
capture its output so a replay never reaches the network.
"""

from __future__ import annotations

import re
import urllib.request
from html import unescape
from html.parser import HTMLParser
from typing import Optional

from ..logging_config import get_logger

log = get_logger("tools.fetch_page")

USER_AGENT = "Mozilla/5.0 (compatible; Maestro-research/0.1; +https://github.com/krishivsaini/Maestro)"
MAX_BYTES = 2_000_000      # stop reading a page past this
MAX_TEXT = 12_000          # extracted text kept per page (and stored in recordings)

_SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "button",
         "select", "iframe", "template"}
_BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "td", "th", "blockquote", "pre", "dd", "dt", "figcaption", "div",
          "section", "article", "br", "tr"}


class _Text(HTMLParser):
    """Readable text, one block per line; drops scripts, navigation and page chrome."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP and self.skip:
            self.skip -= 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def extract_text(html: str) -> str:
    p = _Text()
    try:
        p.feed(html)
        p.close()
    except Exception:  # noqa: BLE001 — malformed HTML: keep what was parsed
        pass
    lines = (" ".join(unescape(line).split()) for line in "".join(p.parts).split("\n"))
    # short lines are menus, buttons and captions, not content
    return "\n".join(line for line in lines if len(line) >= 40)[:MAX_TEXT]


def fetch_text(url: str, timeout: float = 6.0) -> Optional[str]:
    """The readable text of an HTML page, or None if it can't be read."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.5"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if "html" not in ctype.lower():
                return None  # PDFs, images, feeds: the snippet will have to do
            raw = resp.read(MAX_BYTES)
            charset = resp.headers.get_content_charset() or "utf-8"
        text = extract_text(raw.decode(charset, errors="replace"))
        return text or None
    except Exception as exc:  # noqa: BLE001 — blocked, timed out, bad TLS...
        log.info("could not read %s: %s", url, type(exc).__name__)
        return None


_WORD = re.compile(r"[a-z0-9]{3,}")


def passages(text: str, question: str, max_chars: int = 2500) -> str:
    """The paragraphs of ``text`` most relevant to ``question``, in page order, within
    ``max_chars``. Relevance is shared terms, with a bonus for figures (the specifics a
    researcher is most likely to need)."""
    terms = set(_WORD.findall(question.lower()))
    paras = [p for p in text.split("\n") if p.strip()]
    scored = []
    for i, p in enumerate(paras):
        words = set(_WORD.findall(p.lower()))
        score = len(words & terms) + (1 if re.search(r"\d", p) else 0)
        if score:
            scored.append((score, i))
    chosen, used = set(), 0
    for score, i in sorted(scored, key=lambda s: (-s[0], s[1])):
        if used + len(paras[i]) > max_chars and chosen:
            continue
        chosen.add(i)
        used += len(paras[i])
        if used >= max_chars:
            break
    return "\n".join(paras[i][:max_chars] for i in sorted(chosen))
