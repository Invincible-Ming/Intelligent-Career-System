"""Official ima OpenAPI adapter. Only the configured shared library is accessible.

This is document/content search, not a claim that ima exposes a vector/RAG API.
Credentials stay in the trusted backend. No writes, cookies or private API routes.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup

from app.core.config import settings
from app.services.web_fetch import FetchBlocked, _check_url

logger = logging.getLogger(__name__)


class ImaUnavailable(Exception):
    """Safe, actionable error, without upstream credentials or signed URLs."""


def plain_text(value: str) -> str:
    return BeautifulSoup(value, "html.parser").get_text(" ", strip=True)


def relevant_excerpt(text: str, query: str) -> str:
    """Keep the matching part of long originals rather than always their first page."""
    import jieba

    limit = settings.IMA_MAX_CONTENT_CHARS
    if len(text) <= limit:
        return text
    terms = [word.casefold() for word in jieba.lcut(query)
             if len(word.strip()) > 1 and re.search(r"\w", word)]
    folded = text.casefold()
    hits = [folded.find(word) for word in terms]
    positions = [position for position in hits if position >= 0]
    start = max(0, min(positions) - limit // 4) if positions else 0
    return text[start:start + limit]


def validate_media_url(url: str) -> None:
    """Do not turn signed media links into an arbitrary outbound HTTP client."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    trusted = host.endswith((".myqcloud.com", ".qq.com"))
    if parts.scheme != "https" or not trusted or parts.port not in (None, 443):
        raise FetchBlocked("ima 原文地址不属于受信内容域名")
    _check_url(url)


class ImaService:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                trust_env=False, follow_redirects=False, timeout=settings.IMA_TIMEOUT,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    def _headers(self) -> dict[str, str]:
        try:
            client_id = Path(settings.IMA_CLIENT_ID_FILE).expanduser().read_text(encoding="utf-8").strip()
            key = Path(settings.IMA_API_KEY_FILE).expanduser().read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ImaUnavailable("无法读取 ima 凭证文件，请检查后端配置与文件权限") from exc
        if not client_id or not key or any(c in client_id + key for c in "\r\n"):
            raise ImaUnavailable("ima 凭证为空或格式错误")
        return {"ima-openapi-clientid": client_id, "ima-openapi-apikey": key,
                "ima-openapi-ctx": "skill_version=1.1.10"}

    def _library_id(self) -> str:
        if not settings.IMA_KNOWLEDGE_BASE_ID.strip():
            raise ImaUnavailable("尚未配置供系统共享的 ima 知识库")
        return settings.IMA_KNOWLEDGE_BASE_ID.strip()

    async def _post(self, endpoint: str, body: dict, *, note: bool = False) -> dict:
        # The model and caller cannot supply arbitrary endpoints or a base URL.
        allowed = {"get_knowledge_base", "get_knowledge_list", "search_knowledge", "get_media_info"}
        if endpoint not in allowed and not (note and endpoint == "get_doc_content"):
            raise ImaUnavailable("不支持的 ima 操作")
        base = ("https://ima.qq.com/openapi/note/v1" if note else settings.IMA_BASE_URL)
        if base not in {"https://ima.qq.com/openapi/wiki/v1", "https://ima.qq.com/openapi/note/v1"}:
            raise ImaUnavailable("ima 接口地址必须为官方地址")
        try:
            response = await self.client.post(f"{base}/{endpoint}", headers=self._headers(), json=body)
            if response.status_code in (401, 403):
                raise ImaUnavailable("ima 认证或权限校验失败，请检查凭证及知识库权限")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("invalid response")
            code = payload.get("code", payload.get("retcode"))
            if code != 0:
                if code == 220004:
                    raise ImaUnavailable("配置的 ima 知识库 ID 无效，请更新后端 IMA_KNOWLEDGE_BASE_ID")
                if code in (110030, 220030):
                    raise ImaUnavailable("当前 ima 凭证没有读取该资料的权限")
                if code == 110021:
                    raise ImaUnavailable("ima 请求频率受限，请稍后重试")
                raise ImaUnavailable("ima 未能完成请求，请检查知识库状态或稍后重试")
            data = payload.get("data")
            if not isinstance(data, dict):
                raise ValueError("invalid data")
            return data
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise ImaUnavailable("ima 服务暂时不可用，请稍后重试") from exc

    async def library_info(self) -> dict:
        identifier = self._library_id()
        data = await self._post("get_knowledge_base", {"ids": [identifier]})
        infos = data.get("infos") or {}
        # Official docs use id/name; live endpoints also use kb_id/kb_name.
        if isinstance(infos, dict):
            item = infos.get(identifier, {})
        elif isinstance(infos, list):
            item = next((x for x in infos if x.get("kb_id", x.get("id")) == identifier), {})
        else:
            item = {}
        if not isinstance(item, dict) or not item:
            raise ImaUnavailable("ima 未返回配置的知识库，请检查访问权限")
        return {"name": item.get("name", item.get("kb_name", "ima 知识库")),
                "description": item.get("description", "")}

    async def list_items(self, *, cursor: str = "") -> dict:
        data = await self._post("get_knowledge_list", {
            "knowledge_base_id": self._library_id(), "cursor": cursor, "limit": 20,
        })
        return {"items": [{"title": str(x.get("title", x.get("name", "未命名资料")))[:300]}
                          for x in data.get("knowledge_list", []) if isinstance(x, dict)],
                "is_end": bool(data.get("is_end", True)), "next_cursor": data.get("next_cursor", "")}

    async def search(self, *, query: str, owner_id: str, top_k: int) -> list[dict[str, Any]]:
        # A valid application user is mandatory even though this library is shared.
        owner_id = str(uuid.UUID(owner_id))
        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError("检索问题应为 1～2000 字符")
        async with asyncio.timeout(settings.IMA_SEARCH_TIMEOUT):
            return await self._search(query=query, owner_id=owner_id, top_k=max(1, min(top_k, 20)))

    async def _search(self, *, query: str, owner_id: str, top_k: int) -> list[dict]:
        results: list[dict] = []
        seen: set[str] = set()
        cursor = ""
        fetched = 0
        for _ in range(2):
            data = await self._post("search_knowledge", {
                "query": query, "knowledge_base_id": self._library_id(), "cursor": cursor,
            })
            for item in data.get("info_list", [])[:20]:
                media_id = item.get("media_id")
                if not media_id or media_id in seen:
                    continue
                seen.add(media_id)
                title = str(item.get("title", "未命名资料"))[:300]
                content = plain_text(str(item.get("highlight_content") or ""))
                evidence = "search_excerpt"
                if not content and fetched < settings.IMA_MAX_SOURCES:
                    fetched += 1
                    try:
                        content = await self._media_content(media_id=media_id, owner_id=owner_id, query=query)
                        evidence = "original_excerpt"
                    except Exception as exc:
                        logger.info("ima 原文不可读取（%s），跳过无正文资料", type(exc).__name__)
                # A title is not evidence. Never feed an unavailable document to the model as a fact.
                if not content.strip():
                    continue
                rank = len(results) + 1
                results.append({
                    "document_id": f"ima:{media_id}", "document_type": "knowledge",
                    "content": f"资料标题：{title}\n{content[:settings.IMA_MAX_CONTENT_CHARS]}",
                    "score": 1.0 / rank, "source": "ima",
                    "chunk_metadata": {"title": title, "evidence": evidence,
                                       "score_kind": "rank", "shared_knowledge": True},
                })
                if len(results) >= top_k:
                    return results
            next_cursor = data.get("next_cursor", "")
            if data.get("is_end", True) or not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        return results

    async def _media_content(self, *, media_id: str, owner_id: str, query: str) -> str:
        # Called only for media returned by search in the configured library.
        data = await self._post("get_media_info", {"media_id": media_id})
        note_id = (data.get("notebook_ext_info") or {}).get("notebook_id")
        if data.get("media_type") == 11 and note_id:
            note = await self._post("get_doc_content", {"note_id": note_id, "target_content_format": 0}, note=True)
            return relevant_excerpt(str(note.get("content") or ""), query)
        url_info = data.get("url_info") or {}
        url = url_info.get("url", "")
        if not url:
            return ""
        await asyncio.to_thread(validate_media_url, url)
        # A separate request prevents ima API headers from reaching object storage.
        headers = {k: str(v) for k, v in (url_info.get("headers") or {}).items()
                   if k.lower() in {"authorization", "x-cos-security-token"}}
        async with self.client.stream("GET", url, headers=headers) as response:
            if response.status_code != 200:
                return ""
            buffer = bytearray()
            async for block in response.aiter_bytes():
                buffer.extend(block)
                if len(buffer) > settings.max_upload_size_bytes:
                    raise ValueError("ima 原文超过读取大小限制")
        extension = {1: ".pdf", 3: ".docx", 5: ".xlsx", 7: ".md", 13: ".txt"}.get(data.get("media_type"))
        if extension:
            from app.services.sandbox_client import parse_document

            async def no_ocr(*args, **kwargs):
                # Retrieval must not unexpectedly spend a model call on every scanned page.
                return ""

            blocks = await parse_document(owner_id=uuid.UUID(owner_id), run_id=uuid.uuid4(),
                                          data=bytes(buffer), extension=extension, ocr=no_ocr,
                                          max_chars=settings.MAX_PARSED_DOCUMENT_CHARS)
            return relevant_excerpt("\n\n".join(block.text for block in blocks), query)
        if data.get("media_type") in (2, 6, 20):
            soup = BeautifulSoup(bytes(buffer), "html.parser")
            for tag in soup(["script", "style", "noscript", "template"]):
                tag.decompose()
            return relevant_excerpt(soup.get_text("\n", strip=True), query)
        return ""

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


ima_service = ImaService()
