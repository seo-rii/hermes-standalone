"""DuckDuckGo HTML search using HTTPX and the standard-library HTML parser.

The POST fields and result layout follow DDGS 9.16.0's DuckDuckGo engine:
https://github.com/deedy5/ddgs/blob/v9.16.0/ddgs/engines/duckduckgo.py

This backend uses ordinary TLS and the DuckDuckGo HTML endpoint. It does not
reproduce primp's browser TLS fingerprint or DDGS' other search engines;
rate limits and bot challenges are surfaced as search failures. The original
Hermes disposable worker still owns the overall deadline and cancellation.
"""

from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit


SEARCH_URL = "https://html.duckduckgo.com/html/"
_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})


def _target(value):
    if value.startswith("//"):
        value = "https:" + value
    parsed = urlsplit(value)
    if parsed.hostname and (parsed.hostname == "duckduckgo.com" or parsed.hostname.endswith(".duckduckgo.com")):
        if parsed.path == "/y.js":
            return ""
        value = parse_qs(parsed.query).get("uddg", [value])[0]
    elif parsed.path in ("/l/", "/l"):
        value = parse_qs(parsed.query).get("uddg", [""])[0]
    return value if urlsplit(value).scheme in ("https", "http") else ""


class _Results(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.current = None
        self.root = self.title = self.snippet = None
        self.rows = []

    def _finish(self):
        item, self.current = self.current, None
        self.root = self.title = self.snippet = None
        if item and item["href"] and item["title"]:
            self.rows.append({"title": " ".join("".join(item["title"]).split()),
                              "href": item["href"], "body": " ".join("".join(item["body"]).split())})

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        depth = len(self.stack) + 1
        if self.current is None and classes & {"result", "result__body", "body"}:
            self.current = {"title": [], "body": [], "href": ""}
            self.root = depth
        if self.current is not None:
            if tag == "h2":
                self.title = depth
            if tag == "a" and ("result__a" in classes or self.title is not None):
                self.title = self.title or depth
                self.current["href"] = _target(attrs.get("href", ""))
            if classes & {"result__snippet", "snippet"}:
                self.snippet = depth
            if tag == "a" and not self.title and len(self.stack) == self.root and not self.snippet:
                self.snippet = depth
                if not self.current["href"]:
                    self.current["href"] = _target(attrs.get("href", ""))
            if tag == "br":
                self.handle_data(" ")
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        try:
            index = len(self.stack) - 1 - self.stack[::-1].index(tag)
        except ValueError:
            return
        if self.current is not None:
            if self.root is not None and self.root > index:
                self._finish()
            else:
                if self.title is not None and self.title > index:
                    self.title = None
                if self.snippet is not None and self.snippet > index:
                    self.snippet = None
        del self.stack[index:]

    def handle_data(self, data):
        if self.current is not None:
            if self.title is not None:
                self.current["title"].append(data)
            elif self.snippet is not None:
                self.current["body"].append(data)

    def close(self):
        super().close()
        self._finish()


class DDGS:
    def __init__(self, *, timeout=10, endpoint=None):
        import httpx

        self.endpoint = endpoint or SEARCH_URL
        self.client = httpx.Client(timeout=timeout, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
            "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en-US,en;q=0.5",
        })

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.client.close()

    def text(self, query, *, max_results=5):
        wanted = max(1, int(max_results))
        rows, seen = [], set()
        for page in range(1, 11):
            payload = {"q": query, "b": "", "l": "us-en"}
            if page > 1:
                payload["s"] = str(10 + (page - 2) * 15)
            response = self.client.post(self.endpoint, data=payload)
            try:
                response.raise_for_status()
                html = response.text
                if response.status_code == 202 or any(marker in html.lower() for marker in
                        ("anomaly-modal", 'id="challenge-form"', "bots use duckduckgo too")):
                    raise RuntimeError("DuckDuckGo returned a bot challenge or rate limit")
                parser = _Results()
                parser.feed(html)
                parser.close()
            finally:
                response.close()
            added = 0
            for row in parser.rows:
                if row["href"] not in seen:
                    seen.add(row["href"])
                    rows.append(row)
                    added += 1
                    if len(rows) >= wanted:
                        return rows
            if not added:
                break
        return rows
