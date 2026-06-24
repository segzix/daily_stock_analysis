#!/usr/bin/env python3
"""Minimal search proxy: uses curl subprocess (proven working) via proxy."""

import json
import os
import re
import subprocess
import time
from html import unescape
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


def search_google(query: str, max_results: int = 5, time_range: str = "") -> list[dict]:
    proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
    proxy_port = os.getenv("PROXY_PORT", "7897")
    proxy_url = f"http://{proxy_host}:{proxy_port}"

    params = f"q={query}&num={max_results}&hl=zh-CN"
    if time_range == "day":
        params += "&tbs=qdr:d"
    elif time_range == "week":
        params += "&tbs=qdr:w"
    elif time_range == "month":
        params += "&tbs=qdr:m"

    cmd = [
        "curl", "-s", "--max-time", "12",
        "--proxy", proxy_url,
        "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        f"https://www.google.com/search?{params}"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        html = result.stdout
    except Exception:
        return []

    if not html or "<title>Error 429" in html or "captcha" in html.lower():
        return []

    results = []
    for match in re.finditer(
        r'<a[^>]*href="/url\?q=(https?://[^"&]+)[^"]*"[^>]*>'
        r'(?:<[^>]*>)*(.*?)(?:<[^>]*>)*</a>',
        html, re.DOTALL
    ):
        url = match.group(1)
        raw_title = match.group(2)
        title = unescape(re.sub(r"<[^>]+>", "", raw_title).strip())
        if not title or not url or "google.com" in url:
            continue
        results.append({"title": title, "url": url, "content": "", "engine": "google"})
        if len(results) >= max_results:
            break

    return results


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if not self.path.startswith("/search"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return

        # Parse query params
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]
        fmt = params.get("format", ["html"])[0]
        max_results = int(params.get("max_results", ["5"])[0])
        time_range = params.get("time_range", [""])[0]

        if fmt != "json" or not query:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
            return

        results = search_google(query, max_results, time_range)
        body = json.dumps({"query": query, "results": results,
                          "unresponsive_engines": []}, ensure_ascii=False)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())
        self.wfile.flush()

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    port = int(os.getenv("SEARCH_PROXY_PORT", "8888"))
    proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
    proxy_port = os.getenv("PROXY_PORT", "7897")
    os.environ["PROXY_URL"] = f"http://{proxy_host}:{proxy_port}"
    print(f"Search proxy: http://127.0.0.1:{port}  (proxy: {os.environ['PROXY_URL']})", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
