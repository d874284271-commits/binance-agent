from __future__ import annotations

import httpx
import pytest

from binance_agent.intel.content import (
    ContentFetcher,
    document_from_news_item,
    split_text_chunks,
    validate_public_url,
)
from binance_agent.intel.models import NewsItem


class FakePinnedTransport:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def fetch(
        self, url: str, *, resolved_ips: tuple[str, ...], timeout: float, max_bytes: int
    ) -> httpx.Response:
        self.calls.append((url, resolved_ips))
        return self.responses.pop(0)


async def _public_resolver(host: str) -> tuple[str, ...]:
    assert host.endswith("example.com")
    return ("93.184.216.34",)


async def test_content_fetch_uses_pinned_transport_and_injected_extractor() -> None:
    transport = FakePinnedTransport(
        [
            httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b'<p>source</p><a href="https://official.example.com/notice">notice</a>',
            )
        ]
    )
    fetcher = ContentFetcher(
        transport=transport,
        resolver=_public_resolver,
        extractor=lambda html, url: "Extracted body with conditions.",
    )
    document = await fetcher.fetch("https://news.example.com/story", source="News")
    assert document.status == "available"
    assert document.level == "body"
    assert document.text == "Extracted body with conditions."
    assert document.links == ["https://official.example.com/notice"]
    assert transport.calls == [("https://news.example.com/story", ("93.184.216.34",))]


async def test_redirect_target_is_revalidated() -> None:
    transport = FakePinnedTransport(
        [
            httpx.Response(302, headers={"location": "https://news.example.com/final"}),
            httpx.Response(200, headers={"content-type": "text/html"}, content=b"ok"),
        ]
    )
    fetcher = ContentFetcher(
        transport=transport, resolver=_public_resolver, extractor=lambda html, url: "ok"
    )
    document = await fetcher.fetch("https://news.example.com/start")
    assert document.status == "available"
    assert [call[0] for call in transport.calls] == [
        "https://news.example.com/start",
        "https://news.example.com/final",
    ]


async def test_discovery_body_is_media_after_successful_extraction() -> None:
    transport = FakePinnedTransport(
        [httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>body</p>")]
    )
    fetcher = ContentFetcher(
        transport=transport,
        resolver=_public_resolver,
        extractor=lambda html, url: "article body",
    )

    document = await fetcher.fetch(
        "https://news.example.com/story", source="GNews", source_kind="discovery"
    )

    assert document.status == "available"
    assert document.source_kind == "media"


async def test_discovery_body_uses_trusted_kind_for_final_redirect_target() -> None:
    transport = FakePinnedTransport(
        [
            httpx.Response(302, headers={"location": "https://official.example.com/notice"}),
            httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>body</p>"),
        ]
    )
    fetcher = ContentFetcher(
        transport=transport,
        resolver=_public_resolver,
        extractor=lambda html, url: "official body",
        trusted_official_domains={"official.example.com"},
    )

    document = await fetcher.fetch(
        "https://news.example.com/start", source="GNews", source_kind="discovery"
    )

    assert document.status == "available"
    assert document.url == "https://official.example.com/notice"
    assert document.source_kind == "official"


async def test_network_fetch_refuses_unpinned_default_transport() -> None:
    fetcher = ContentFetcher(resolver=_public_resolver, extractor=lambda html, url: "text")
    document = await fetcher.fetch("https://news.example.com/story")
    assert document.status == "unavailable"
    assert document.fetch_error == "safe_transport_required"
    assert document.source_kind == "unknown"


async def test_dns_safety_failure_keeps_stable_error_code() -> None:
    async def unsafe_resolver(host: str) -> tuple[str, ...]:
        raise ValueError("non_public_dns_result")

    fetcher = ContentFetcher(resolver=unsafe_resolver)
    document = await fetcher.fetch("https://news.example.com/story")
    assert document.status == "failed"
    assert document.fetch_error == "non_public_dns_result"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/a",
        "http://[::1]/a",
        "file:///etc/passwd",
        "http://user@news.example.com/a",
    ],
)
def test_public_url_rejects_private_or_unsafe_addresses(url: str) -> None:
    with pytest.raises(ValueError):
        validate_public_url(url)


def test_provider_text_becomes_summary_evidence_without_network() -> None:
    item = NewsItem(title="Headline", summary="Provider supplied summary", source="RSS")
    document = document_from_news_item(item)
    assert document.level == "summary"
    assert document.text == "Provider supplied summary"


def test_long_text_chunks_keep_overlap_and_explicit_truncation() -> None:
    text = "A" * 7000 + "\n\n" + "B" * 7000 + "\n\n" + "C" * 7000
    chunks = split_text_chunks(text, max_chars=8000, overlap_chars=500, max_chunks=2)
    assert len(chunks) == 2
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
    assert chunks[-1]["truncated"] is True
    assert int(chunks[1]["start"]) < int(chunks[0]["end"])


async def test_httpx_transport_pins_ip_but_keeps_tls_name_and_host() -> None:
    from binance_agent.intel.content import HttpxPinnedTransport

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "news.example.com"
        assert request.extensions["sni_hostname"] == "news.example.com"
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<p>body</p>")

    transport = HttpxPinnedTransport(lambda: httpx.MockTransport(handler))
    response = await transport.fetch(
        "https://news.example.com/article",
        resolved_ips=("93.184.216.34",),
        timeout=1,
        max_bytes=100,
    )
    assert response.text == "<p>body</p>"
    with pytest.raises(ValueError, match="content_too_large"):
        await transport.fetch(
            "https://news.example.com/article",
            resolved_ips=("93.184.216.34",),
            timeout=1,
            max_bytes=2,
        )


def test_real_trafilatura_extracts_article_not_navigation() -> None:
    from binance_agent.intel.content import trafilatura_extractor

    body = "The issuer submitted an application. The regulator has not approved it. Conditions apply in France only. "
    html = (
        "<html><head><title>Application status</title></head><body><nav>HOME LOGIN</nav><article><h1>Application status</h1>"
        + "".join("<p>" + body + "</p>" for _ in range(8))
        + "</article></body></html>"
    )
    text = trafilatura_extractor(html, "https://example.com/notice")
    assert text and "has not approved" in text
    assert "HOME LOGIN" not in text


@pytest.mark.parametrize(
    "text,max_chars,overlap,max_chunks",
    [
        ("  short body  ", 20, 2, 1),
        ("a" * 150, 80, 10, 2),
        ("a" * 151, 80, 10, 3),
        ("para one\n\npara two\n\nlast paragraph", 20, 3, 8),
    ],
)
def test_complete_chunks_have_exact_offsets_and_no_false_truncation(
    text: str,
    max_chars: int,
    overlap: int,
    max_chunks: int,
) -> None:
    chunks = split_text_chunks(
        text, max_chars=max_chars, overlap_chars=overlap, max_chunks=max_chunks
    )
    assert chunks
    assert not chunks[-1].get("truncated")
    assert chunks[-1]["end"] == len(text)
    assert all(chunk["text"] == text[int(chunk["start"]) : int(chunk["end"])] for chunk in chunks)
