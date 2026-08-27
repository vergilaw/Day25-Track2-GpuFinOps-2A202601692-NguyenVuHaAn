"""A production-grade LiteLLM-style token-cost tracker & governance proxy with:
- Per-API-key hard budget caps (HARD-STOP before overrun)
- Soft budget warnings (e.g. at 80% of budget cap)
- Intelligent model auto-downgrade (cascade to small model under budget pressure)
- Request tagging (team, project, env) & chargeback-ready FOCUS export
- Real-time spend metrics & unit economics ($/1M-token)

Deck §10 "Token Tier" of cost observability: attribute $/request and prevent bill shock.
"""
from __future__ import annotations
import os
import sys
import time
from collections import defaultdict
from typing import Callable

# Add project root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from finops import pricing, allocation

MODEL_PRICES = {  # $/1M tokens: (input_price, output_price)
    "small": (0.20, 0.40),
    "large": (3.00, 15.00),
}


class BudgetExceeded(Exception):
    """Raised when a request would breach the key's hard USD budget cap."""
    pass


class CostTracker:
    def __init__(
        self,
        budgets: dict[str, float] | None = None,
        warning_threshold: float = 0.80,
        warning_callback: Callable[[str, float, float], None] | None = None,
        auto_downgrade_on_warning: bool = False,
    ):
        """Initialize the cost tracker proxy.

        Args:
            budgets: Mapping of api_key -> max USD budget cap.
            warning_threshold: Fraction of budget (0..1) to trigger soft warning (default 80%).
            warning_callback: Optional hook called as callback(api_key, current_spend, budget_cap).
            auto_downgrade_on_warning: If True, automatically routes 'large' model requests
                                       to 'small' model once budget exceeds warning_threshold.
        """
        self.budgets = budgets or {}
        self.spend: dict[str, float] = defaultdict(float)
        self.tokens_served: dict[str, int] = defaultdict(int)
        self.warning_threshold = warning_threshold
        self.warning_callback = warning_callback
        self.auto_downgrade_on_warning = auto_downgrade_on_warning
        self.warnings_issued: set[str] = set()
        self.log: list[dict] = []

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic estimation of tokens (~4 characters per token)."""
        return max(1, len(text) // 4)

    def complete(
        self,
        api_key: str,
        model: str,
        prompt: str,
        max_output_tokens: int = 256,
        cached_input_tokens: int = 0,
        batch: bool = False,
        team: str | None = None,
        project: str | None = None,
        environment: str = "prod",
    ) -> dict:
        """Process an LLM completion request with cost tracking and budget governance."""
        in_tok = self._estimate_tokens(prompt)
        out_tok = max_output_tokens
        cap = self.budgets.get(api_key)

        effective_model = model
        # Feature: Auto-downgrade to 'small' model if key is under budget pressure (>80%)
        if cap is not None and self.auto_downgrade_on_warning:
            current_ratio = self.spend[api_key] / cap if cap > 0 else 0
            if current_ratio >= self.warning_threshold and model == "large":
                effective_model = "small"

        pin, pout = MODEL_PRICES[effective_model]
        cost = pricing.request_cost(
            in_tok,
            out_tok,
            pin,
            pout,
            cached_in=cached_input_tokens,
            batch=batch,
        )

        # Check hard budget cap
        if cap is not None and self.spend[api_key] + cost > cap:
            raise BudgetExceeded(
                f"BLOCKED: key '{api_key}' would spend ${self.spend[api_key] + cost:.4f} > cap ${cap:.2f}"
            )

        # Update telemetry
        self.spend[api_key] += cost
        total_req_tokens = in_tok + out_tok
        self.tokens_served[api_key] += total_req_tokens

        # Check soft budget warning threshold (e.g. 80%)
        if cap is not None:
            spend_ratio = self.spend[api_key] / cap
            if spend_ratio >= self.warning_threshold and api_key not in self.warnings_issued:
                self.warnings_issued.add(api_key)
                if self.warning_callback:
                    self.warning_callback(api_key, self.spend[api_key], cap)

        rec = {
            "ts": time.strftime("%H:%M:%S"),
            "key": api_key,
            "team": team or api_key,
            "project": project or "default",
            "environment": environment,
            "requested_model": model,
            "model": effective_model,
            "downgraded": (model != effective_model),
            "in": in_tok,
            "cached_in": cached_input_tokens,
            "out": out_tok,
            "total_tokens": total_req_tokens,
            "batch": batch,
            "cost": round(cost, 6),
            "spend_so_far": round(self.spend[api_key], 6),
        }
        rec["text"] = f"[mock {effective_model}] ok ({in_tok} in / {out_tok} out)"
        self.log.append(rec)
        return rec

    def report(self) -> dict:
        """Return spending summary per API key."""
        return {k: round(v, 4) for k, v in self.spend.items()}

    def detailed_summary(self) -> dict:
        """Return detailed governance metrics per key."""
        out = {}
        for k, spent in self.spend.items():
            cap = self.budgets.get(k)
            toks = self.tokens_served[k]
            dollars_per_m = pricing.dollars_per_million(spent, toks)
            out[k] = {
                "spend_usd": round(spent, 4),
                "budget_usd": round(cap, 4) if cap is not None else None,
                "utilization_pct": round((spent / cap) * 100, 1) if cap else None,
                "tokens_served": toks,
                "dollars_per_1m_tokens": round(dollars_per_m, 3),
                "requests_count": sum(1 for r in self.log if r["key"] == k),
                "downgraded_count": sum(1 for r in self.log if r["key"] == k and r["downgraded"]),
            }
        return out

    def to_focus_export(self, billing_account: str = "nimbusai-litellm") -> list[dict]:
        """Convert recorded proxy logs to standard FinOps FOCUS 1.0 schema rows."""
        return allocation.to_focus_rows(self.log, billing_account=billing_account)
