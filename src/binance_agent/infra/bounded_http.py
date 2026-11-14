"""Bounded discovery HTTP reads, including servers ignoring identity encoding."""

from __future__ import annotations

import zlib
from typing import Any

import httpx

MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class ResponseTooLarge(httpx.HTTPError):
    def __init__(self) -> None:
        super().__init__("response_too_large")


class BoundedAsyncClient(httpx.AsyncClient):
    """兼容 AsyncClient 的 GET 客户端,对传输体和解码体分别限制为 4 MiB。

    其他请求方法保持 httpx 原有语义;GET 返回前读完并关闭响应,超限、解压或网络错误向上传递。
    """

    async def get(self, url: Any, **kwargs: Any) -> httpx.Response:
        """逐跳执行 GET 并有界读取最终响应;沿用 httpx 的跨来源凭证剥离和重定向次数上限。"""
        headers = httpx.Headers(kwargs.pop("headers", None))
        headers["Accept-Encoding"] = "identity"
        follow_redirects = kwargs.pop("follow_redirects", self.follow_redirects)
        auth = kwargs.pop("auth", httpx.USE_CLIENT_DEFAULT)
        request = self.build_request("GET", url, headers=headers, **kwargs)
        history: list[httpx.Response] = []
        while True:
            # Native follow_redirects reads intermediate bodies without our cap.
            response = await self.send(request, auth=auth, stream=True, follow_redirects=False)
            try:
                if follow_redirects and response.next_request is not None:
                    if len(history) >= self.max_redirects:
                        raise httpx.TooManyRedirects("redirect_limit", request=request)
                    history.append(response)
                    request = response.next_request
                    # httpx prepared next_request, including cross-origin auth stripping.
                    auth = None
                    continue
                result = await self._bounded_response(response)
                result.history = history
                return result
            finally:
                await response.aclose()

    @staticmethod
    async def _bounded_response(response: httpx.Response) -> httpx.Response:
        """验证体积及编码,增量解码 gzip/deflate;返回完整响应,超限或压缩流不完整时抛错。"""
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_RESPONSE_BYTES:
            raise ResponseTooLarge()
        encoding = response.headers.get("content-encoding", "identity").lower().strip()
        if encoding not in {"identity", "", "gzip", "deflate"}:
            raise httpx.DecodingError("unsupported_content_encoding")
        decoder = (
            zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS)
            if encoding in {"gzip", "deflate"}
            else None
        )
        body = bytearray()
        wire_size = 0
        # MockTransport may supply an already-materialized Response. Network
        # responses always take aiter_raw; avoid decoding mock content twice.
        if response.is_stream_consumed:
            content = response.content
            if len(content) > MAX_RESPONSE_BYTES:
                raise ResponseTooLarge()
            return response
        async for chunk in response.aiter_raw():
            wire_size += len(chunk)
            if wire_size > MAX_RESPONSE_BYTES:
                raise ResponseTooLarge()
            try:
                decoded = (
                    decoder.decompress(chunk, MAX_RESPONSE_BYTES - len(body) + 1)
                    if decoder is not None
                    else chunk
                )
            except zlib.error as exc:
                raise httpx.DecodingError("invalid_compressed_response") from exc
            body.extend(decoded)
            if len(body) > MAX_RESPONSE_BYTES or (decoder and decoder.unconsumed_tail):
                raise ResponseTooLarge()
        if decoder is not None and (not decoder.eof or decoder.unused_data):
            raise httpx.DecodingError("invalid_compressed_response")
        result_headers = response.headers.copy()
        result_headers.pop("content-encoding", None)
        result_headers.pop("content-length", None)
        return httpx.Response(
            response.status_code,
            headers=result_headers,
            content=bytes(body),
            request=response.request,
            extensions=response.extensions,
            history=response.history,
        )
