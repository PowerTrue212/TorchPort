"""Small, auditable web search/fetch backend for controlled Agent trials."""

from __future__ import annotations

import html
import ipaddress
import json
import socket
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Any

MAX_DOWNLOAD_BYTES = 1_000_000
MAX_TEXT_CHARS = 20_000
USER_AGENT = "TorchPort-Harness/2.0"


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def _public_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("只允许公开的 http/https URL")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port)}
    except socket.gaierror as exc:
        raise ValueError(f"无法解析 URL 主机：{parsed.hostname}") from exc
    if any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise PermissionError("禁止访问本机、内网或保留地址")
    return value


def _open(url: str, *, timeout: float = 30.0) -> tuple[str, bytes, str]:
    request = urllib.request.Request(_public_url(url), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        final_url = _public_url(response.geturl())
        content_type = response.headers.get_content_type()
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("网页超过 1 MB 限制")
    return final_url, data, content_type


def search(query: str, *, limit: int = 8) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("搜索词不能为空")
    results = []
    github_query = " ".join(
        word for word in query.split() if word.casefold() not in {"github", "repository", "repo"}
    ) or query
    github_url = (
        "https://api.github.com/search/repositories?per_page="
        f"{min(limit, 10)}&q={urllib.parse.quote(github_query)}"
    )
    try:
        _, github_data, _ = _open(github_url)
        for item in json.loads(github_data).get("items", []):
            results.append(
                {
                    "title": item.get("full_name") or "",
                    "url": item.get("html_url") or "",
                    "snippet": item.get("description") or "",
                    "source": "github",
                }
            )
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    url = "https://www.bing.com/search?format=rss&q=" + urllib.parse.quote(query)
    _, data, _ = _open(url)
    root = ET.fromstring(data)
    seen = {item["url"] for item in results}
    for item in root.findall("./channel/item"):
        item_url = item.findtext("link") or ""
        if not item_url or item_url in seen:
            continue
        results.append(
            {
                "title": html.unescape(item.findtext("title") or ""),
                "url": item_url,
                "snippet": html.unescape(item.findtext("description") or ""),
                "source": "bing",
            }
        )
        seen.add(item_url)
        if len(results) >= limit:
            break
    return {"query": query, "results": results[:limit]}


def fetch(url: str) -> dict[str, Any]:
    final_url, data, content_type = _open(url)
    charset = "utf-8"
    text = data.decode(charset, "replace")
    if content_type in {"text/html", "application/xhtml+xml"}:
        parser = _Text()
        parser.feed(text)
        text = "\n".join(parser.parts)
    text = text[:MAX_TEXT_CHARS]
    return {
        "url": final_url,
        "content_type": content_type,
        "text": text,
        "truncated": len(text) == MAX_TEXT_CHARS,
    }
