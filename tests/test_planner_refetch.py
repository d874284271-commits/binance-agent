"""Planner 不再拥有 refetch;补抓行为由 test_data_preflight.py 覆盖。"""

from __future__ import annotations

from binance_agent.agents.nodes import StrategyPlannerAction, _coerce_planner_action
from binance_agent.agents.prompts import STRATEGY_ACTION_PLANNER


def test_coerce_planner_action_rejects_legacy_refetch_actions() -> None:
    for name in ("refetch_intel", "refetch_market"):
        action, violation = _coerce_planner_action({"action": name, "reason": "补数据"})

        assert isinstance(action, StrategyPlannerAction)
        assert action.action == "watchlist"
        assert violation is not None
        assert "非法 planner action" in violation


def test_strategy_planner_prompt_assigns_refetch_to_preflight() -> None:
    assert "data_preflight" in STRATEGY_ACTION_PLANNER
    assert '"action":"no_trade|watchlist|continue_to_synth|rerun_quality_check"' in (
        STRATEGY_ACTION_PLANNER
    )
