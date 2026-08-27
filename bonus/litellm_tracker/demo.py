"""Demo: LiteLLM token-cost tracker with budget governance, alerts, cascade, and FOCUS export.

Run: python demo.py (or from root: python bonus/litellm_tracker/demo.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracker import CostTracker, BudgetExceeded


def on_warning(key: str, current: float, cap: float):
    print(f"  [ALERT] Soft limit reached for '{key}': ${current:.4f} / ${cap:.2f} ({current/cap*100:.1f}%)")


def run_demo():
    print("=" * 65)
    print("  BONUS: LiteLLM Cost Tracker & FinOps Budget Governance")
    print("=" * 65)

    # 1. Initialize tracker with per-key budgets
    tracker = CostTracker(
        budgets={
            "team-chat": 0.05,        # strict low budget for chatbot
            "team-cascade": 0.05,     # auto-downgrades to small model when budget strained
            "team-eval": 10.00,       # high batch budget
        },
        warning_threshold=0.80,
        warning_callback=on_warning,
        auto_downgrade_on_warning=True,
    )

    print("\n--- Scenario 1: team-chat (Without auto-downgrade -> hits Hard Block) ---")
    chat_tracker = CostTracker(
        budgets={"team-chat": 0.05},
        warning_threshold=0.80,
        warning_callback=on_warning,
        auto_downgrade_on_warning=False,
    )
    for i in range(1, 25):
        try:
            chat_tracker.complete(
                api_key="team-chat",
                model="large",
                prompt="Summarize this very long customer transcript " * 25,
                max_output_tokens=200,
                team="chat-team",
                project="customer-support",
            )
        except BudgetExceeded as e:
            print(f"  [HARD-STOP] Request #{i} BLOCKED: {e}")
            break

    print("\n--- Scenario 2: team-cascade (With auto-downgrade -> saves budget & keeps serving) ---")
    for i in range(1, 25):
        try:
            r = tracker.complete(
                api_key="team-cascade",
                model="large",
                prompt="Analyze this text query " * 25,
                max_output_tokens=200,
                team="nlp-team",
                project="cascade-poc",
            )
            if r["downgraded"]:
                print(f"  [CASCADE] Req #{i}: Downgraded large -> small (Cost: ${r['cost']:.6f}, Spend: ${r['spend_so_far']:.4f})")
        except BudgetExceeded as e:
            print(f"  [HARD-STOP] Req #{i} BLOCKED: {e}")
            break

    print("\n--- Scenario 3: team-eval (Batch + Prompt Caching) ---")
    for i in range(10):
        tracker.complete(
            api_key="team-eval",
            model="small",
            prompt="Evaluate sentiment of benchmark review " * 10,
            cached_input_tokens=50,
            batch=True,
            team="eval-team",
            project="daily-evals",
        )
    print("  Completed 10 batched + cached eval requests successfully.")

    # 4. Summary & Governance Report
    print("\n" + "=" * 65)
    print("  GOVERNANCE & UNIT ECONOMICS SUMMARY")
    print("=" * 65)
    details = tracker.detailed_summary()
    for key, d in details.items():
        print(f"Key: {key:<14} | Spend: ${d['spend_usd']:<7.4f} / ${d['budget_usd'] or 0:<6.2f} "
              f"({d['utilization_pct'] or 0:>5.1f}%) | "
              f"Tokens: {d['tokens_served']:<6,} | Unit Econ: ${d['dollars_per_1m_tokens']:<6.3f}/1M | "
              f"Reqs: {d['requests_count']} (Downgraded: {d['downgraded_count']})")

    # 5. FOCUS 1.0 Export
    focus_rows = tracker.to_focus_export()
    print(f"\nFOCUS Export generated: {len(focus_rows)} rows ready for cloud billing ingestion.")
    print(f"Sample FOCUS row: {focus_rows[0]}")


if __name__ == "__main__":
    run_demo()
