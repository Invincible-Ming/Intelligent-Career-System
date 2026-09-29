"""受限网页抓取：SSRF 防护 + 重定向逐跳校验 + 大小/超时限制 + 正文提取。

仅供 Agent 的网页阅读工具使用；抓取地址可能来自不可信的用户输入，
因此按"不可信外部内容"处理：只允许公网 http/https、禁止携带凭据、
逐跳重校验解析结果、限制大小与超时。
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

MAX_REDIRECTS = 3
MAX_BYTES = 2_000_000
FETCH_TIMEOUT = 15.0
ALLOWED_PORTS = {80, 443}
ALLOWED_SCHEMES = {"http", "https"}
USER_AGENT = "CareerSystem-Agent/1.0 (personal job-assistant; document reader)"


class FetchBlocked(Exception):
    """地址或内容触发了抓取限制。"""


def _check_url(url: str) -> None:
    """校验 URL：协议、端口、凭据，并确认解析出的 IP 均为公网地址。"""
    parts = urlsplit(url)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise FetchBlocked("仅支持 http/https 地址")
    if parts.port is not None and parts.port not in ALLOWED_PORTS:
        raise FetchBlocked("仅允许 80/443 端口")
    if parts.username or parts.password:
        raise FetchBlocked("不允许携带凭据的地址")
    host = parts.hostname
    if not host:
        raise FetchBlocked("地址无效")
    try:
        infos = socket.getaddrinfo(
            host, parts.port or (443 if parts.scheme == "https" else 80),
            proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchBlocked("域名无法解析") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise FetchBlocked("禁止访问内网或保留地址")


async def fetch_page_text(url: str, *, max_chars: int = 8000) -> str:
    """抓取网页正文（去脚本/样式），超长截断。重定向逐跳重新校验。"""
    _check_url(url)
    current = url
    async with httpx.AsyncClient(
        follow_redirects=False, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT},
    ) as client:
        text = ""
        title = ""
        links: list[str] = []
        for _ in range(MAX_REDIRECTS):
            _check_url(current)
            async with client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchBlocked("重定向缺少目标地址")
                    current = str(resp.url.join(location))
                    continue
                if resp.status_code >= 400:
                    raise FetchBlocked(f"网页返回错误状态 {resp.status_code}")
                content_type = resp.headers.get("content-type", "").split(";")[0].strip()
                if not content_type.startswith(("text/", "application/json", "application/xml")):
                    raise FetchBlocked(f"不支持的内容类型：{content_type}")
                buffer = bytearray()
                async for chunk in resp.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) >= MAX_BYTES:
                        break
                html = bytes(buffer[:MAX_BYTES]).decode(resp.encoding or "utf-8", errors="replace")

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "template", "svg"]):
                tag.decompose()
            title = (soup.title.get_text(strip=True) if soup.title else "")[:120]
            base = str(resp.url) if resp else current
            for anchor in soup.find_all("a", href=True):
                if len(links) >= 15:
                    break
                absolute = urljoin(base, anchor["href"])
                parts = urlsplit(absolute)
                if parts.scheme in ALLOWED_SCHEMES and parts.hostname:
                    label = anchor.get_text(strip=True)[:60]
                    links.append(f"- {label or absolute}: {absolute}")
            body = soup.body or soup
            lines = [ln.strip() for ln in body.get_text("\n").splitlines()]
            text = "\n".join(ln for ln in lines if ln)
            break
        else:
            raise FetchBlocked("重定向次数过多")

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[页面内容过长，已截断]"
    parts_out = []
    if title:
        parts_out.append(f"页面标题：{title}")
    parts_out.append(text or "(页面无可提取正文)")
    if links:
        parts_out.append("页面链接：\n" + "\n".join(links))
    return "\n\n".join(parts_out)
