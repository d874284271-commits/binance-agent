"""Read-only report and export endpoints, including a fully offline example."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from ..agents.reports import AssetTarget, ReportResult, market_report, report_markdown


def example_report() -> ReportResult:
    return ReportResult(
        kind="market",
        title="示例研报 · 如何阅读结论",
        target=AssetTarget(symbol="BTCUSDT", market="fapi"),
        analysis_at="合成示例,无实时行情",
        summary="先保持观察,等待趋势与资料相互印证。",
        recommendations=["示例展示如何把结论、条件与证据放在一起;不代表当前市场观点。"],
        conditions=["新的已核验资料或市场条件发生变化时,更新研报重新评估。"],
        risks=["示例中的内容全部为合成,不能用于交易决策。"],
        basis=[
            {
                "statement": "真实研报会在此展示影响判断的已核验事实。",
                "quote": "合成示例文字,没有引用真实网页。",
            }
        ],
        gaps=["没有实时行情、新闻或模型调用。"],
        completeness="insufficient_evidence",
        synthetic=True,
    )


def saved_report(store: Any, run_id: str) -> ReportResult:
    record = store.get(run_id)
    if record is None:
        raise HTTPException(404, "找不到这份研究")
    if record.get("status") != "completed":
        raise HTTPException(409, "本轮尚未完成,请等待或恢复研究")
    if record.get("report"):
        return ReportResult.model_validate(record["report"])
    if record.get("final_advice"):
        return market_report(record, record["final_advice"])
    raise HTTPException(404, "旧记录没有可读取的报告")


def register_report_routes(app: FastAPI) -> None:
    @app.get("/reports/example")
    async def example() -> dict[str, Any]:
        return {"report": example_report().model_dump(mode="json")}

    @app.get("/runs/{run_id}/report")
    async def report(run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "report": saved_report(app.state.run_store, run_id).model_dump(mode="json"),
        }

    @app.get("/runs/{run_id}/report.md", response_class=PlainTextResponse)
    async def markdown(run_id: str) -> PlainTextResponse:
        return PlainTextResponse(
            report_markdown(saved_report(app.state.run_store, run_id)),
            media_type="text/markdown; charset=utf-8",
        )
