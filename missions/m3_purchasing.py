"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py

EXTENSION 1 is wired in here: the mission scores the DOCUMENTED simple policy
against the economic policy (per-GPU reclaim risk + 1yr/3yr term + the fact that a
reservation bills 24/7) and prints the delta between them.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def _simple_policy_cost(job, cat) -> tuple[str, float]:
    """The original lab policy: duty-cycle heuristic, reserved billed on used hours."""
    g = job["gpu_type"]
    c = cat[g]
    gpu_hours = num(job["hours_per_day"]) * DAYS * int(num(job["num_gpus"]))
    od = num(c["on_demand_hr"])
    tier = pricing.recommend_tier(num(job["hours_per_day"]),
                                  bool(int(num(job["interruptible"]))))
    if tier == "spot":
        cost = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
    elif tier == "reserved":
        cost = gpu_hours * num(c["reserved_3yr_hr"])   # <- bills only hours USED
    else:
        cost = gpu_hours * od
    return tier, cost


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = simple_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        job_days = num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # --- EXTENSION 1: economic tier choice, priced per useful hour ---
        d = pricing.recommend_tier_detailed(
            hpd, interruptible, gpu_type=gtype, job_days=job_days, prices=c)
        opt_cost = d["effective_hr"] * gpu_hours

        # --- baseline policy, for the before/after comparison ---
        simple_tier, simple_cost = _simple_policy_cost(j, cat)

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        simple_monthly += simple_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": d["tier"],
                     "term": d["term"], "simple_tier": simple_tier,
                     "interrupt_rate": d["interrupt_rate"],
                     "effective_hr": d["effective_hr"], "reason": d["reason"],
                     "candidates": d["candidates"],
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost),
                     "simple_optimized": round(simple_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    simple_pct = ((on_demand_monthly - simple_monthly) / on_demand_monthly * 100
                  if on_demand_monthly else 0.0)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'term':6}{'$/useful-h':>11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}{str(r['term'] or '-'):6}"
                  f"{r['effective_hr']:>11.3f}"
                  f"{'$' + format(r['on_demand'], ','):>12}"
                  f"{'$' + format(r['optimized'], ','):>12}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- EXTENSION 1: simple duty-cycle policy vs economic policy --")
        print(f"  simple policy   : ${simple_monthly:,.0f}/mo  ({simple_pct:.1f}% saved)")
        print(f"  economic policy : ${optimized_monthly:,.0f}/mo  ({savings_pct:.1f}% saved)")
        print(f"  delta           : ${optimized_monthly - simple_monthly:+,.0f}/mo "
              f"({savings_pct - simple_pct:+.1f} pts)")
        print("  The simple policy looked cheaper only because it billed reservations for")
        print("  hours USED. A commitment bills 24/7, so its 'savings' were overstated:")
        for r in recs:
            if r["optimized"] != r["simple_optimized"]:
                print(f"    {r['job_id']:18} ${r['simple_optimized']:>7,} -> ${r['optimized']:>7,} "
                      f"({r['simple_tier']} -> {r['tier']})")
        print("\n  per-GPU reclaim risk vs the rate at which spot would stop winning:")
        print(f"    {'gpu':8}{'spot$':>7}{'od$':>7}{'actual':>9}{'break-even':>12}")
        for g in sorted({r["gpu_type"] for r in recs}):
            c = cat[g]
            be = pricing.break_even_interrupt_rate(num(c["spot_hr"]), num(c["on_demand_hr"]))
            print(f"    {g:8}{num(c['spot_hr']):>7.2f}{num(c['on_demand_hr']):>7.2f}"
                  f"{pricing.spot_interrupt_rate(g):>8.0%}/h{be:>11.0%}/h")
        print("    -> every break-even is >100%/h: spot wins for interruptible work even in")
        print("       a hostile pool, so reclaim risk is not the reason to avoid it.")
        print("\n  why each job landed where it did:")
        for r in recs:
            print(f"    {r['job_id']:18} {r['tier']:10} {r['reason']}")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "simple_policy_monthly": round(simple_monthly),
            "simple_policy_savings_pct": round(simple_pct, 1)}


if __name__ == "__main__":
    run()
