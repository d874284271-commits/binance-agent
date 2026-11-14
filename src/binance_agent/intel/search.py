"""Optional Tavily discovery. Results are leads, never verified evidence."""

from __future__ import annotations

from typing import Any

import httpx

from ..infra.bounded_http import BoundedAsyncClient


class TavilySearch:
    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self.client = client
        self.api_key = api_key

    async def __call__(self, query: str) -> list[dict[str, Any]]:
        request = self.client.build_request(
            "POST",
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.api_key}", "Accept-Encoding": "identity"},
            json={
                "query": query[:500],
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=12,
        )
        response = await self.client.send(request, stream=True, follow_redirects=False)
        try:
            if response.status_code != 200:
                raise ValueError(f"搜索服务暂不可用 (HTTP {response.status_code})")
            bounded = await BoundedAsyncClient._bounded_response(response)
            data = bounded.json()
            return [
                {"url": row["url"], "title": str(row.get("title") or ""), "source": "Tavily"}
                for row in data.get("results", [])[:5]
                if isinstance(row, dict) and isinstance(row.get("url"), str)
            ]
        finally:
            await response.aclose()
