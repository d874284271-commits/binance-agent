"""角色输入投影:保留来源状态,仅压缩明确属于模型论述的文本。"""

from __future__ import annotations

import copy
import json
from typing import Any

from .context import CompressibleBlock, ContextEnvelope


def news_evidence_view(research: dict[str, Any]) -> dict[str, Any]:
    """生成单一事实表及来源引用,不重复携带任务全文或文章正文。

    去重保留事实限定条件、来源与支持状态,争议或不同来源的断言不能因合并
    而升级;只在投影内解析别名,原始 research 及其保存 ID 不变。
    """
    view = {
        key: copy.deepcopy(research[key])
        for key in (
            "version",
            "question",
            "analysis_at",
            "critical_unresolved",
            "gaps",
            "conflicts",
            "document_coverage",
            "reading_complete",
            "evidence_validation",
            "legacy_no_original_evidence",
            "search_status",
            "input_mode",
            "workflow",
            "sentiment_features",
        )
        if key in research
    }
    facts: list[dict[str, Any]] = []
    identities: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for index, raw in enumerate(research.get("facts") or []):
        if not isinstance(raw, dict):
            continue
        fact = copy.deepcopy(raw)
        old_id = str(fact.get("fact_id") or f"fact-{index}")
        # citation_id is derived from fact_id, not evidence identity. Compare
        # the original document/quote/offset/status instead, retaining all IDs
        # in saved records and resolving downstream fact references via aliases.
        factual_identity = {k: v for k, v in fact.items() if k not in {"fact_id", "task_id"}}
        if isinstance(factual_identity.get("citations"), list):
            factual_identity["citations"] = [
                {k: v for k, v in citation.items() if k != "citation_id"}
                if isinstance(citation, dict)
                else citation
                for citation in factual_identity["citations"]
            ]
        identity = json.dumps(
            factual_identity,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if identity not in identities:
            identities[identity] = old_id
            fact["fact_id"] = old_id
            facts.append(fact)
        aliases[old_id] = identities[identity]
    view["facts"] = facts
    sources: dict[str, dict[str, Any]] = {}
    for document in research.get("documents", []):
        if isinstance(document, dict) and document.get("evidence_id"):
            sources[str(document["evidence_id"])] = {
                k: copy.deepcopy(document[k])
                for k in (
                    "evidence_id",
                    "content_sha256",
                    "url",
                    "source",
                    "source_kind",
                    "title",
                    "published_at",
                    "fetched_at",
                    "extracted_dates",
                    "level",
                    "status",
                    "fetch_error",
                )
                if k in document
            }
    view["sources"] = list(sources.values())
    view["tasks"] = [
        {
            **{
                k: copy.deepcopy(task[k])
                for k in (
                    "task_id",
                    "question",
                    "completion",
                    "critical",
                    "status",
                    "gaps",
                    "conflicts",
                )
                if k in task
            },
            "fact_ids": list(
                dict.fromkeys(
                    aliases[str(f["fact_id"])]
                    for f in task.get("facts", [])
                    if str(f.get("fact_id")) in aliases
                )
            ),
        }
        for task in research.get("tasks", [])
        if isinstance(task, dict)
    ]
    # Claim bindings retain their status, but factual text and quotations live
    # exclusively in the fact table. Unbound legacy claims keep their full data.
    view["claims"] = []
    for raw in research.get("claims", []):
        claim = copy.deepcopy(raw)
        ids = claim.get("fact_ids") or []
        if ids and all(str(i) in aliases for i in ids):
            claim = {
                k: claim[k]
                for k in ("claim_id", "claim_type", "event_id", "market_paths", "support")
                if k in claim
            }
            claim["fact_ids"] = list(dict.fromkeys(aliases[str(i)] for i in ids))
        view["claims"].append(claim)
    return view


def narrative_envelope(payload: dict[str, Any]) -> ContextEnvelope:
    """把研究总结和辩论论述移到带来源标签的可压缩块,其余载荷深拷贝保留。

    结构化 claim、风险闸门、引文及证据记录保持原值;不修改调用方 payload,
    也不把压缩后的模型论述当作新的证据。
    """
    protected = copy.deepcopy(payload)
    blocks: list[CompressibleBlock] = []
    summary = protected.pop("research_summary", None)
    if summary:
        blocks.append(CompressibleBlock(source_id="research_summary", text=str(summary)))
        protected["research_summary_context_id"] = "research_summary"
    for field in ("debate", "debate_turns", "prior_round_turns"):
        for turn in protected.get(field, []):
            argument = turn.pop("argument", "")
            if argument:
                source_id = f"debate:{turn['side']}:{turn['round']}:argument"
                blocks.append(CompressibleBlock(source_id=source_id, text=str(argument)))
                turn["argument_context_id"] = source_id
    return ContextEnvelope(protected_data=protected, compressible_blocks=blocks)
