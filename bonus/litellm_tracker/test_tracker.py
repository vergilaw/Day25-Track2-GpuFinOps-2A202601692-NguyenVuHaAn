"""Tests for the LiteLLM Cost Tracker bonus module."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracker import CostTracker, BudgetExceeded


def test_tracker_records_spend_and_tokens():
    t = CostTracker()
    r = t.complete("key-1", "small", "Hello world", max_output_tokens=50)
    assert r["cost"] > 0
    assert t.spend["key-1"] > 0
    assert t.tokens_served["key-1"] > 0
    assert len(t.log) == 1


def test_tracker_enforces_hard_budget():
    t = CostTracker(budgets={"key-limited": 0.001})
    # First small request passes
    t.complete("key-limited", "small", "hi", max_output_tokens=10)
    # Huge request breaches budget and is blocked
    with pytest.raises(BudgetExceeded):
        t.complete("key-limited", "large", "prompt " * 200, max_output_tokens=500)


def test_tracker_soft_warning_and_auto_downgrade():
    warnings = []
    t = CostTracker(
        budgets={"key-warn": 0.005},
        warning_threshold=0.50,
        warning_callback=lambda k, s, b: warnings.append(k),
        auto_downgrade_on_warning=True,
    )
    # Spend enough to pass 50% threshold ($0.0025)
    # 2 requests of large model with max_output_tokens=150 costs ~ 2 * $0.0023 = $0.0046 > $0.0025
    t.complete("key-warn", "large", "test prompt " * 20, max_output_tokens=150)
    t.complete("key-warn", "large", "test prompt " * 20, max_output_tokens=150)
    assert "key-warn" in warnings

    # Next large request should be auto-downgraded to small model
    r = t.complete("key-warn", "large", "test", max_output_tokens=20)
    assert r["requested_model"] == "large"
    assert r["model"] == "small"
    assert r["downgraded"] is True


def test_tracker_focus_export():
    t = CostTracker()
    t.complete("key-team", "small", "prompt", max_output_tokens=20, team="core", project="test")
    rows = t.to_focus_export()
    assert len(rows) == 1
    assert rows[0]["BillingAccountId"] == "nimbusai-litellm"
    assert rows[0]["Tags"]["team"] == "core"
    assert rows[0]["Tags"]["project"] == "test"
