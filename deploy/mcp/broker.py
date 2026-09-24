"""A fixed-endpoint search broker, reachable only via a mounted Unix socket."""
import asyncio
import http.client
import json
import os
import socket
import ssl
from html.parser import HTMLParser
from urllib.parse import urlencode

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from policy import SEARCH_HOST, result_url, validate_query


class SearchResults(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.item = None
        self.depth = 0
        self.heading = False
        self.paragraph = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "li" and "b_algo" in attrs.get("class", "").split():
            self.item = {"title": "", "snippet": "", "url": ""}
            self.depth = 1
        elif self.item is not None:
            if tag == "li":
                self.depth += 1
            if tag == "h2":
                self.heading = True
            if tag == "p":
                self.paragraph = True
            if tag == "a" and self.heading and not self.item["url"]:
                self.item["url"] = result_url(attrs.get("href", "")) or ""

    def handle_endtag(self, tag):
        if self.item is None:
            return
        if tag == "h2":
            self.heading = False
        if tag == "p":
            self.paragraph = False
        if tag == "li":
            self.depth -= 1
            if self.depth == 0:
                if self.item["url"] and self.item["title"].strip():
                    self.results.append(
                        {k: " ".join(v.split())[:(600 if k == "snippet" else 1500)] for k, v in self.item.items()})
                self.item = None
                self.heading = self.paragraph = False

    def handle_data(self, data):
        if self.item is not None:
            if self.heading:
                self.item["title"] += data
            if self.paragraph:
                self.item["snippet"] += data


def fetch_search(query, count):
    # The firewall and this socket share the same startup-resolved public IPs.
    # HTTPSConnection retains Bing's hostname for TLS verification and Host.
    addresses = os.environ["MCP_SEARCH_ADDRESSES"].split(",")
    html = None
    for address in addresses:
        connection = http.client.HTTPSConnection(SEARCH_HOST, timeout=15, context=ssl.create_default_context())
        connection._create_connection = lambda destination, timeout, source_address=None,
                                               ip=address: socket.create_connection((ip, 443), timeout)
        try:
            connection.request("GET", "/search?" + urlencode({"q": query, "count": count}),
                               headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html",
                                        "Accept-Encoding": "identity"})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError("搜索服务拒绝请求；不跟随重定向")
            body = response.read(1024 * 1024 + 1)
            if len(body) > 1024 * 1024 or "text/html" not in response.getheader("Content-Type", ""):
                raise ValueError("搜索响应超限或格式不支持")
            html = body.decode("utf-8", errors="replace")
            break
        except (OSError, ValueError, http.client.HTTPException):
            continue
        finally:
            connection.close()
    if html is None:
        raise ValueError("公开搜索暂不可用，请稍后重试")
    parser = SearchResults()
    parser.feed(html)
    if not parser.results:
        raise ValueError("搜索未返回可解析的结果，可能遇到验证码；不扩大网络权限")
    return {"results": parser.results[:count], "untrusted_content": True}


slots = asyncio.Semaphore(4)


async def search(request: Request):
    try:
        body = await request.body()
        if len(body) > 4096:
            return JSONResponse({"error": "请求过大"}, status_code=413)
        payload = json.loads(body)
        if not isinstance(payload, dict) or set(payload) - {"query", "count"}:
            raise ValueError("仅接受搜索词与结果数量，不接受 URL")
        count = payload.get("count", 5)
        query = validate_query(payload.get("query", ""), count)
    except (ValueError, TypeError, AttributeError):
        return JSONResponse({"error": "非法搜索参数"}, status_code=400)
    try:
        async with slots:
            return JSONResponse(await asyncio.to_thread(fetch_search, query, count))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)


app = Starlette(routes=[Route("/search", search, methods=["POST"])])

if __name__ == "__main__":
    os.chmod("/run/search", 0o700)
    try:
        os.unlink("/run/search/search.sock")
    except FileNotFoundError:
        pass
    uvicorn.run(app, uds="/run/search/search.sock", log_level="warning", access_log=False)
