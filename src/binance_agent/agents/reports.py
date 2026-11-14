"""Saved product reports shared by the reader, history and Markdown export."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .state import DISCLAIMER, StrategyAdvice


class AssetTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["asset"] = "asset"
    symbol: str = Field(min_length=2, max_length=40, pattern=r"^[A-Za-z0-9]+$")
    market: Literal["api", "fapi"]

    @field_validator("symbol")
    @classmethod
    def upper(cls, value: str) -> str:
        return value.upper()


class IndependentTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["independent"] = "independent"


ResearchTarget = Annotated[AssetTarget | IndependentTarget, Field(discriminator="kind")]


class ReportResult(BaseModel):
    version: Literal[1] = 1
    kind: Literal["market", "topic", "legacy"]
    title: str
    target: ResearchTarget | None = None
    analysis_at: str
    question: str = ""
    input_revision: int = 0
    summary: str
    recommendations: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    basis: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    coverage: list[dict[str, str]] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    completeness: Literal["complete", "partial", "insufficient_evidence"]
    advice: StrategyAdvice | None = None
    synthetic: bool = False
    disclaimer: str = DISCLAIMER


def report_evidence(research: dict[str, Any]) -> dict[str, Any]:
    """Only expose facts that passed both citation and semantic checks."""
    facts = [
        f for f in research.get("facts", []) if f.get("support") in {"supported", "model_supported"}
    ]
    ids = {f.get("evidence_id") for f in facts}
    return {
        "basis": [
            {k: f.get(k) for k in ("statement", "evidence_id", "quote", "scope", "uncertainty")}
            for f in facts
        ],
        "sources": [
            {
                k: d.get(k)
                for k in (
                    "evidence_id",
                    "title",
                    "url",
                    "source",
                    "published_at",
                    "fetched_at",
                    "level",
                )
            }
            for d in research.get("documents", [])
            if d.get("evidence_id") in ids
        ],
    }


def market_report(record: dict[str, Any], advice: Any) -> ReportResult:
    """Organize already gated results; never call a model to rewrite the report."""
    value = (
        advice
        if isinstance(advice, StrategyAdvice)
        else StrategyAdvice.model_validate(
            {"symbol": record.get("symbol"), "decision_type": "no_trade", **advice}
        )
    )
    research = record.get("news_research") or {}
    evidence = report_evidence(research)
    gaps = list(dict.fromkeys(value.evidence_gaps))
    return ReportResult(
        kind="market" if record.get("mode") == "report" else "legacy",
        title=f"{value.symbol} 市场研报"
        if record.get("mode") == "report"
        else f"{value.symbol} 历史研究",
        target=record.get("target"),
        analysis_at=record.get("analysis_at")
        or research.get("analysis_at")
        or record.get("created_at")
        or "",
        question=(record.get("research_input") or {}).get("research_question", ""),
        input_revision=record.get("input_revision", 0),
        summary=value.rationale,
        recommendations=[value.no_trade_reason] if value.no_trade_reason else [],
        conditions=value.reentry_conditions,
        risks=value.risk_warnings,
        gaps=gaps,
        completeness="insufficient_evidence"
        if not evidence["basis"]
        else "partial"
        if gaps or not research.get("reading_complete")
        else "complete",
        coverage=[
            {"source": source, "status": status}
            for source, status in (record.get("intel_snapshot") or {})
            .get("source_status", {})
            .items()
        ],
        advice=value,
        synthetic=bool(research.get("synthetic")),
        **evidence,
    )


def incomplete_report(record: dict[str, Any], reason: str) -> ReportResult:
    return ReportResult(
        kind="topic",
        title=record.get("title") or "专题研究",
        target=record.get("target"),
        analysis_at=record.get("analysis_at") or record.get("created_at") or "",
        input_revision=record.get("input_revision", 0),
        question=(record.get("research_input") or {}).get("research_question", ""),
        summary=reason,
        gaps=[reason],
        completeness="insufficient_evidence",
    )


def report_markdown(report: ReportResult) -> str:
    """Export the selected immutable report, excluding runtime and secret fields."""

    def text(value: Any) -> str:
        # Markdown can itself contain executable HTML when read in other tools.
        return str(value or "").replace("<", "&lt;").replace(">", "&gt;")

    scope = "独立专题"
    if isinstance(report.target, AssetTarget):
        scope = f"{report.target.symbol} · {'现货' if report.target.market == 'api' else '合约'}"
    elif report.target is None:
        scope = "历史研究(原版本未记录研究意图)"
    lines = [
        f"# {text(report.title)}",
        "",
        f"范围:{scope}",
        f"分析时点:{text(report.analysis_at)}",
        f"完整性:{ {'complete': '完整', 'partial': '部分完成', 'insufficient_evidence': '证据不足'}[report.completeness] }",
    ]
    if report.synthetic:
        lines += ["", "合成示例,仅展示界面,不代表真实市场或模型效果。"]
    lines += ["", "## 直接结论", "", text(report.summary)]
    for title, rows in (
        ("建议与条件", report.recommendations + report.conditions),
        ("重要风险", report.risks),
    ):
        if rows:
            lines += ["", f"## {title}", ""] + [f"- {text(row)}" for row in rows]
    if report.advice and report.advice.decision_type == "trade":
        a = report.advice
        lines += [
            "",
            "## 条件性价位",
            "",
            f"方向:{a.direction}",
            f"入场:{text(a.entry_zone)}",
            f"止损:{text(a.stop_loss)}",
            f"目标:{text(' / '.join(a.targets))}",
        ]
    lines += ["", "## 主要依据", ""]
    for fact in report.basis:
        lines += [f"- {text(fact.get('statement'))}"]
        if fact.get("quote"):
            lines += [f"  原文:{text(fact['quote'])}"]
    lines += ["", "## 来源与缺口", ""]
    for source in report.sources:
        lines += [
            f"- {text(source.get('source'))} · {text(source.get('title'))}",
            f"  {text(source.get('url'))}",
            f"  发布时间:{text(source.get('published_at')) or '未知'};获取时间:{text(source.get('fetched_at'))}",
        ]
    lines += [f"- {text(gap)}" for gap in report.gaps]
    if report.coverage:
        lines += ["", "资料覆盖:"] + [
            f"- {text(row.get('source'))}: {text(row.get('status'))}" for row in report.coverage
        ]
    return "\n".join([*lines, "", text(report.disclaimer), ""])
