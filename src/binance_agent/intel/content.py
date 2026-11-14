"""通过可注入的固定 IP 传输抓取公共 HTML 证据。

仅做 DNS 预检再交普通 HTTP client 连接,仍可能遭 DNS rebinding;
因此抓取要求 PinnedContentTransport 将实际连接固定到已经验证的公共 IP。
ContentFetcher 逐跳检查重定向,传输层禁用自动重定向并限制响应体大小。
"""

from __future__ import annotations

import asyncio
import importlib.util
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from datetime import datetime
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import urljoin, urlsplit

import httpx

from .evidence import evidence_document
from .models import EvidenceDocument, NewsItem


class PinnedContentTransport(Protocol):
    """实际 TCP 连接必须使用 resolved_ips 中的已验证地址,并保留正确 TLS/SNI。

    传输实现应流式限制 max_bytes、禁用自动重定向;调用方在每次跳转后重新
    验证 URL 和 DNS,不能由底层 client 静默跟随未经检查的目标。
    """

    async def fetch(
        self, url: str, *, resolved_ips: tuple[str, ...], timeout: float, max_bytes: int
    ) -> httpx.Response: ...


Resolver = Callable[[str], Awaitable[tuple[str, ...]]]
Extractor = Callable[[str, str], str | None]


def split_text_chunks(
    text: str, *, max_chars: int = 8000, overlap_chars: int = 500, max_chunks: int = 8
) -> list[dict[str, int | str]]:
    """按段落边界切分原文,返回字符半开偏移及文本,不修改原文空白。

    max_chars 与 overlap_chars 的单位为字符;达到 max_chunks 后尚有剩余正文
    时在最后块标记 truncated,调用方必须据此保留覆盖缺口。
    """
    if max_chars <= 0 or overlap_chars < 0 or overlap_chars >= max_chars or max_chunks < 1:
        raise ValueError("invalid text chunk limits")
    source = text
    if not source.strip():
        return []
    chunks: list[dict[str, int | str]] = []
    cursor = 0
    length = len(source)
    while cursor < length and len(chunks) < max_chunks:
        target = min(length, cursor + max_chars)
        end = target
        if target < length:
            boundary = max(source.rfind("\n\n", cursor, target), source.rfind("\n", cursor, target))
            if boundary > cursor + max_chars // 2:
                end = boundary
        chunk = source[cursor:end]
        if chunk:
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "start": cursor,
                    "end": end,
                    "text": chunk,
                }
            )
        if end >= length:
            break
        next_cursor = max(cursor + 1, end - overlap_chars)
        cursor = next_cursor
    if chunks and int(chunks[-1]["end"]) < length:
        chunks[-1]["truncated"] = True
    return chunks


class ContentFetcher:
    def __init__(
        self,
        *,
        transport: PinnedContentTransport | None = None,
        resolver: Resolver | None = None,
        extractor: Extractor | None = None,
        timeout: float = 8.0,
        max_bytes: int = 1_500_000,
        max_redirects: int = 3,
        trusted_official_domains: set[str] | None = None,
    ) -> None:
        if timeout <= 0 or max_bytes <= 0 or max_redirects < 0:
            raise ValueError("invalid content fetch limits")
        self._transport = transport
        self._resolver = resolver or resolve_public_ips
        self._extractor = extractor or trafilatura_extractor
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects
        self._trusted_official_domains = {
            domain.casefold().lstrip(".") for domain in (trusted_official_domains or set())
        }

    async def fetch(self, url: str, **kwargs: object) -> EvidenceDocument:
        """在包含解析、跳转、下载和抽取的总超时内抓取正文,超时返回 unavailable。

        kwargs 提供来源与已知发布时间,不推断日期;任务取消不转为普通失败文档。
        """
        try:
            return await asyncio.wait_for(self._fetch(url, **kwargs), timeout=self._timeout)  # type: ignore[arg-type]
        except (TimeoutError, asyncio.TimeoutError):
            return _unavailable(url, "", "", None, "unknown", "timeout")

    async def _fetch(
        self,
        url: str,
        *,
        source: str = "",
        title: str = "",
        published_at: datetime | None = None,
        source_kind: str = "unknown",
    ) -> EvidenceDocument:
        """逐跳校验公共 URL 与 DNS,经固定 IP 传输取得 HTML 并抽取正文。

        不支持的内容、受限页面、超大响应或抽取失败保留明确状态,不能当作空正文成功。
        只返回稳定错误类别,不将完整请求异常或响应内容带入证据错误字段。
        """
        current = url
        try:
            for _ in range(self._max_redirects + 1):
                host = validate_public_url(current)
                ips = await self._resolver(host)
                resolved_kind = self._source_kind(current, source_kind)
                if not ips:
                    return _unavailable(
                        current, source, title, published_at, resolved_kind, "dns_unavailable"
                    )
                if self._transport is None:
                    return _unavailable(
                        current,
                        source,
                        title,
                        published_at,
                        resolved_kind,
                        "safe_transport_required",
                    )
                response = await self._transport.fetch(
                    current, resolved_ips=ips, timeout=self._timeout, max_bytes=self._max_bytes
                )
                location = response.headers.get("location")
                if response.is_redirect and location:
                    current = urljoin(current, location)
                    continue
                if response.status_code != 200:
                    return _failed(
                        current,
                        source,
                        title,
                        published_at,
                        resolved_kind,
                        f"HTTP {response.status_code}",
                    )
                content_type = response.headers.get("content-type", "").lower()
                if not _is_html(content_type):
                    return _unavailable(
                        current,
                        source,
                        title,
                        published_at,
                        resolved_kind,
                        "unsupported_content_type",
                    )
                declared_size = response.headers.get("content-length")
                if (
                    declared_size
                    and declared_size.isdigit()
                    and int(declared_size) > self._max_bytes
                ):
                    return _unavailable(
                        current, source, title, published_at, resolved_kind, "content_too_large"
                    )
                raw = response.content
                if len(raw) > self._max_bytes:
                    return _unavailable(
                        current, source, title, published_at, resolved_kind, "content_too_large"
                    )
                html = raw.decode(response.encoding or "utf-8", errors="replace")
                if re.search(r'isAccessibleForFree["\']?\s*:\s*false', html, re.I) or re.search(
                    r'<input[^>]+type\s*=\s*["\']password', html, re.I
                ):
                    return _unavailable(
                        current, source, title, published_at, resolved_kind, "access_restricted"
                    )
                try:
                    text = await asyncio.to_thread(self._extractor, html, current)
                except Exception:
                    # Cancellation is a BaseException and must propagate.
                    return _unavailable(
                        current, source, title, published_at, resolved_kind, "extractor_failed"
                    )
                if not text:
                    return _unavailable(
                        current,
                        source,
                        title,
                        published_at,
                        resolved_kind,
                        "extractor_unavailable_or_empty",
                    )
                links = await extract_public_links(html, current, resolver=self._resolver)
                return evidence_document(
                    url=current,
                    source=source,
                    title=title,
                    text=text,
                    links=links,
                    published_at=published_at,
                    level="body",
                    # A discovery API only establishes that a candidate URL
                    # exists.  Once the page body has been captured, classify
                    # that evidence from the final URL: trusted domains stay
                    # official; other successfully read pages become media.
                    source_kind=self._source_kind(current, source_kind, body_available=True),
                )
            return _unavailable(
                current,
                source,
                title,
                published_at,
                self._source_kind(current, source_kind),
                "redirect_limit",
            )
        except (ValueError, OSError, httpx.HTTPError, TimeoutError, asyncio.TimeoutError) as exc:
            # Do not copy a response body or full request exception into evidence.
            return _failed(
                current,
                source,
                title,
                published_at,
                self._source_kind(current, source_kind),
                _stable_fetch_error(exc),
            )

    def _source_kind(self, url: str, requested: str, *, body_available: bool = False) -> str:
        """按最终 URL 与正文是否成功读取确定证据来源类型。

        受信域名为 official;成功提取的其他正文为 media,失败或待抓取的搜索结果
        仍可保持 discovery,避免把发现线索当成已经读取的媒体证据。
        """
        host = urlsplit(url).hostname or ""
        if _is_trusted_domain(host, self._trusted_official_domains):
            return "official"
        if body_available:
            return "media"
        return (
            requested if requested in {"media", "community", "discovery", "unknown"} else "unknown"
        )


def document_from_news_item(item: NewsItem, *, prefer_summary: bool = True) -> EvidenceDocument:
    """仅用来源已返回的标题或摘要建立证据,不访问网络、不冒充正文。"""
    text = item.summary.strip() if prefer_summary and item.summary.strip() else item.title
    level = "summary" if text == item.summary.strip() and text else "title"
    return evidence_document(
        url=item.url,
        source=item.source,
        title=item.title,
        text=text,
        published_at=item.published_at,
        fetched_at=item.fetched_at,
        level=level,
        source_kind=item.source_kind,
    )


async def resolve_public_ips(host: str) -> tuple[str, ...]:
    """解析主机并只接受全部为全球可路由地址的 DNS 结果。

    公私地址混合也拒绝,防止后续连接从同批结果选到私网地址;返回去重排序的 IP。
    """
    rows = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    ips = tuple(sorted({row[4][0] for row in rows}))
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ValueError("non_public_dns_result")
    return ips


def validate_public_url(url: str) -> str:
    """校验 HTTP(S)、主机、凭证及端口,返回去除末尾点的主机名。

    仅允许默认端口或显式 80/443,并拒绝非公共 IP 字面量;域名是否解析到公共
    地址仍需 resolve_public_ips,不能凭此函数单独建立抓取授权。
    """
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("public_http_url_required")
    if parsed.username or parsed.password or parsed.port not in {None, 80, 443}:
        raise ValueError("unsafe_url")
    host = parsed.hostname.rstrip(".")
    try:
        if not ipaddress.ip_address(host).is_global:
            raise ValueError("non_public_ip")
    except ValueError as exc:
        if str(exc) == "non_public_ip":
            raise
    return host


def trafilatura_extractor(html: str, url: str) -> str | None:
    """用 Trafilatura 提取正文,排除评论和表格;依赖不存在或无正文返回 None。"""
    if importlib.util.find_spec("trafilatura") is None:
        return None
    import trafilatura

    result = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
    return result if isinstance(result, str) else None


async def extract_public_links(
    html: str, base_url: str, *, resolver: Resolver, limit: int = 20
) -> list[str]:
    """解析并逐项筛选可继续研究的链接,返回去重后的 URL 线索。

    这里只校验 URL 形状,不进行 DNS 或正文抓取;resolver 参数保留给既有调用方,
    实际 DNS 检查在抓取时进行,返回链接不能直接当作已核验证据。
    """
    parser = _LinkParser()
    parser.feed(html)
    links: list[str] = []
    for href in parser.links:
        candidate = urljoin(base_url, href)
        if candidate in links:
            continue
        try:
            validate_public_url(candidate)
        except (ValueError, OSError):
            continue
        links.append(candidate)
        if len(links) >= limit:
            break
    return links


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        for key, value in attrs:
            if key.casefold() == "href" and value:
                self.links.append(value)
                return


def _is_html(content_type: str) -> bool:
    return content_type.startswith("text/html") or content_type.startswith("application/xhtml+xml")


def _stable_fetch_error(exc: BaseException) -> str:
    known = {
        "public_http_url_required",
        "unsafe_url",
        "non_public_ip",
        "non_public_dns_result",
        "safe_transport_required",
        "encoded_body_not_supported",
        "content_too_large",
    }
    message = str(exc)
    return message if message in known else type(exc).__name__


def _is_trusted_domain(host: str, domains: set[str]) -> bool:
    normalized = host.casefold().rstrip(".")
    return any(normalized == domain or normalized.endswith("." + domain) for domain in domains)


def _unavailable(
    url: str, source: str, title: str, published_at: datetime | None, source_kind: str, error: str
) -> EvidenceDocument:
    document = evidence_document(
        url=url,
        source=source,
        title=title,
        published_at=published_at,
        level="body",
        source_kind=source_kind,
    )
    return document.model_copy(update={"status": "unavailable", "fetch_error": error})


def _failed(
    url: str, source: str, title: str, published_at: datetime | None, source_kind: str, error: str
) -> EvidenceDocument:
    document = evidence_document(
        url=url,
        source=source,
        title=title,
        published_at=published_at,
        level="body",
        source_kind=source_kind,
    )
    return document.model_copy(update={"status": "failed", "fetch_error": error})


class HttpxPinnedTransport:
    """把 TCP 连接固定到已验证 IP,同时保留原 Host 与 TLS 主机名。

    每次请求独占连接池,避免跨域 IP 连接复用;抓取不读取环境代理。
    """

    def __init__(
        self, transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None
    ) -> None:
        self._factory = transport_factory or (lambda: httpx.AsyncHTTPTransport(retries=0))

    async def fetch(
        self, url: str, *, resolved_ips: tuple[str, ...], timeout: float, max_bytes: int
    ) -> httpx.Response:
        """用首个已验证公共 IP 发出单次 GET,返回未自动跳转的响应。

        流式读取超过 max_bytes 时失败,拒绝编码响应;连接池在返回或异常时关闭,
        响应 request 恢复原 URL 供上层处理,不把固定 IP 当作证据来源。
        """
        host = validate_public_url(url)
        if not resolved_ips or any(not ipaddress.ip_address(ip).is_global for ip in resolved_ips):
            raise ValueError("non_public_dns_result")
        original = httpx.URL(url)
        pinned = original.copy_with(host=resolved_ips[0])
        async with (
            httpx.AsyncClient(
                transport=self._factory(), trust_env=False, follow_redirects=False
            ) as client,
            client.stream(
                "GET",
                pinned,
                headers={
                    "Host": original.netloc.decode(),
                    "User-Agent": "binance-agent/0.2 research",
                    "Accept-Encoding": "identity",
                },
                extensions={"sni_hostname": host},
                timeout=timeout,
            ) as response,
        ):
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise ValueError("encoded_body_not_supported")
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ValueError("content_too_large")
            headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() not in {"content-encoding", "content-length"}
            }
            return httpx.Response(
                response.status_code,
                headers=headers,
                content=bytes(data),
                request=httpx.Request("GET", url),
            )
