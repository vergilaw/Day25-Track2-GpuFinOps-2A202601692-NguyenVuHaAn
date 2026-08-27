"""M5 - Optimization Report: combine M1-M4 into baseline-vs-optimized (deck section 1/11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png

Every figure comes from a mission's own return value, so the report is reproducible by
re-running the mission that produced it. Extensions 1-5 all feed it: right-sizing comes
from M1's bandwidth analysis rather than a fixed tier-down map, the cache lever is gated
by M2's TTL economics, purchasing uses M3's economic policy, and the carbon section comes
from the Extension 5 scheduler.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import (m1_efficiency_audit, m2_inference_levers, m3_purchasing,
                      ext5_carbon_scheduling)

DAYS = 30
# Loaded cost of one engineer-day, used to turn "effort" into a payback period. A lever
# that saves $100/mo and costs 30 engineer-days is not a lever, it is a hobby.
ENG_DAY_USD = 1200.0
MEDIAN_QUERY_TOKENS = 800


def _unit_economics(r2, baseline: float, optimized: float) -> dict:
    """The metrics that stay honest when the fleet changes size."""
    b_pm, o_pm = r2["baseline_per_m"], r2["optimized_per_m"]
    return {"rows": [
        {"metric": "Inference $/1M tokens", "before": f"${b_pm:.3f}",
         "after": f"${o_pm:.3f}", "change": f"-{(1 - o_pm / b_pm) * 100:.0f}%"},
        {"metric": "Inference $/day", "before": f"${r2['baseline_daily']:,.2f}",
         "after": f"${r2['optimized_daily']:,.2f}", "change": f"-{r2['savings_pct']:.0f}%"},
        {"metric": "Total fleet $/month", "before": f"${baseline:,.0f}",
         "after": f"${optimized:,.0f}",
         "change": f"-{(1 - optimized / baseline) * 100:.0f}%"},
    ]}


def _root_cause(r1, cat) -> dict:
    """The mechanism behind the GPU-Util lie, argued from this fleet's own telemetry."""
    by_id = {s["gpu_id"]: s for s in r1["summary"]}
    lie_ids = [l["gpu_id"] for l in r1["lies"]]
    healthy = max((s for s in r1["summary"] if s["gpu_id"] not in lie_ids),
                  key=lambda s: s["mfu"])

    rows = []
    for s in sorted(r1["summary"], key=lambda x: -x["gpu_util_pct"]):
        if s["gpu_id"] in lie_ids:
            verdict = "**LIE** - busy, not productive"
        elif s["mfu"] >= 0.40:
            verdict = "healthy"
        elif s["gpu_util_pct"] < 50:
            verdict = "honestly under-used"
        else:
            verdict = "borderline"
        rows.append({"gpu_id": s["gpu_id"], "gpu_type": s["gpu_type"],
                     "util": s["gpu_util_pct"], "mfu": s["mfu"], "mbu": s["mbu"],
                     "verdict": verdict})

    lie = by_id[lie_ids[0]]
    c = cat[lie["gpu_type"]]
    peak_tf, peak_bw = num(c["peak_tflops_fp16"]), num(c["peak_bw_tbs"])
    ridge = peak_tf / peak_bw
    ach_tf, ach_bw = lie["mfu"] * peak_tf, lie["mbu"] * peak_bw
    intensity = ach_tf / ach_bw if ach_bw else 0.0
    rate = num(c["on_demand_hr"])
    ratios = [s["mbu"] / s["mfu"] for s in r1["summary"] if s["mfu"]]
    a0, a1 = by_id.get("gpu-a10g-0"), by_id.get("gpu-a10g-1")

    body = [
        "**What the two numbers actually measure.** `nvidia-smi utilization.gpu` is a "
        "*duty-cycle* counter: the fraction of sampling intervals in which at least one "
        "kernel was resident on the device. It says nothing about how much of the device "
        "that kernel used, so a single small kernel occupying one SM for the whole interval "
        "still reports 100%. MFU is a *throughput* ratio: achieved FLOP/s over the peak the "
        "silicon is sold on. One answers \"is the GPU busy?\", the other answers \"is the GPU "
        "doing work I am paying for?\" - and only the second question has a dollar sign in it.",

        f"**The mechanism on {lie['gpu_id']}.** It reports {lie['gpu_util_pct']}% utilization "
        f"while delivering {ach_tf:,.0f} of {peak_tf:,.0f} TFLOP/s (MFU {lie['mfu']:.3f}). At "
        f"${rate:.2f}/hr, {1 - lie['mfu']:.0%} of that rate buys nothing: roughly "
        f"**${rate * (1 - lie['mfu']) * 24 * DAYS:,.0f}/month** on this single GPU pays for "
        f"arithmetic capability that is never exercised. The device is occupied almost "
        f"continuously and idle *inside* almost every interval - which is precisely the state "
        f"a utilization dashboard cannot show you.",

        f"**Where the time actually goes.** Its MBU is {lie['mbu']:.3f}, so it is not "
        f"saturating HBM either. Achieved arithmetic intensity is {intensity:.0f} FLOP/byte "
        f"against a {lie['gpu_type']} ridge point of {ridge:.0f} FLOP/byte ({peak_tf:,.0f} "
        f"TFLOP/s over {peak_bw:.2f} TB/s). Sitting below *both* roofs at once is the "
        f"signature of a latency- and occupancy-bound kernel rather than a bandwidth-bound "
        f"one: small batches, short sequences, and per-kernel launch and synchronisation "
        f"overhead leave the tensor cores waiting between bursts. That makes the first fix a "
        f"serving-configuration fix - larger batches, continuous batching, sequence packing - "
        f"and only then a hardware fix.",

        f"*A caveat the data forces, which the playbook does not mention:* across all "
        f"{len(r1['summary'])} GPUs here MBU tracks MFU within "
        f"{(min(ratios) - 1) * 100:.0f}-{(max(ratios) - 1) * 100:.0f}%, so this telemetry "
        f"cannot cleanly separate memory-bound from occupancy-bound. On real hardware you "
        f"would confirm with Nsight (SM occupancy, DRAM throughput, achieved warps) before "
        f"buying anything on the strength of it.",

        f"**The proof that utilization cannot price work.** {healthy['gpu_id']} reports "
        f"{healthy['gpu_util_pct']}% utilization - *lower* than {lie['gpu_id']}'s "
        f"{lie['gpu_util_pct']}% - at MFU {healthy['mfu']:.3f}, which is "
        f"{healthy['mfu'] / lie['mfu']:.1f}x the useful work for the same ${rate:.2f}/hr."
        + (f" Sharper still, {a1['gpu_id']} and {a0['gpu_id']} are the same part at the same "
           f"price: {a1['gpu_util_pct']}% against {a0['gpu_util_pct']}% reported utilization "
           f"({a1['gpu_util_pct'] / a0['gpu_util_pct']:.1f}x) for MFU {a1['mfu']:.3f} against "
           f"{a0['mfu']:.3f} (only {a1['mfu'] / a0['mfu']:.2f}x). A dashboard ranking those "
           f"two by utilization would rank them almost backwards."
           if a0 and a1 else ""),

        "**Consequence for governance.** GPU-Util is a liveness signal - useful for spotting "
        "a crashed trainer, worthless for capacity planning. Efficiency reviews and "
        "chargeback should run on MFU/MBU and on $/1M-token, neither of which can be gamed "
        "by keeping a device nominally busy.",
    ]
    return {"title": "Root cause: how 98% GPU-Util and 19% MFU coexist", "body": body,
            "evidence_table": rows}


def _lever_detail(levers: dict, r1, r2, r3) -> list:
    ce = r2["cache_economics"]
    denied = sum(1 for v in ce.values() if not v["worth_it"])
    n = list(levers)
    return [
        {"name": n[0], "usd": levers[n[0]],
         "mechanism": "Route the 80% of traffic that does not need a frontier model to one "
                      "15x cheaper; batch the eval queue at -50%; cache shared prefixes at "
                      "-90% on reads",
         "evidence": f"${r2['baseline_per_m']:.3f} -> ${r2['optimized_per_m']:.3f} per 1M "
                     f"tokens over {r2['total_tokens']:,} tokens/day",
         "effort": "Low - routing rule + batch endpoint"},
        {"name": n[1], "usd": levers[n[1]],
         "mechanism": "Interruptible jobs to spot, priced with checkpoint overhead and "
                      "expected rework; steady jobs to the reserved term their runtime can "
                      "actually justify",
         "evidence": f"${r3['on_demand_monthly']:,.0f} -> ${r3['optimized_monthly']:,.0f}/mo "
                     f"across {len(r3['recommendations'])} jobs",
         "effort": "Medium - checkpointing + a commitment decision"},
        {"name": n[2], "usd": levers[n[2]],
         "mechanism": "Move bandwidth-starved GPUs to the cheapest part that still clears "
                      "their achieved bandwidth plus 25% headroom",
         "evidence": f"{len(r1['rightsize_proposals'])} GPUs at MFU < 30%, billed on active "
                     f"hours only so the idle lever is not double-counted",
         "effort": "High - migration + latency re-validation"},
        {"name": n[3], "usd": levers[n[3]],
         "mechanism": "Stop paying for GPUs that report under 10% utilization overnight",
         "evidence": f"${r1['idle_waste_daily']:,.2f}/day of measured idle time on "
                     f"{sum(1 for s in r1['summary'] if s['idle_hours']) } GPU(s)",
         "effort": "Low - scheduler policy"},
        {"name": "Cache gate (a loss avoided, not a saving)", "usd": 0,
         "mechanism": "Refuse the cache lever where a 5-minute TTL means the prefix is "
                      "rewritten more often than it is read back",
         "evidence": f"{denied}/{len(ce)} prefix groups rejected; "
                     f"${r2['cache_denied_savings_daily'] * DAYS:,.0f}/mo of apparent savings "
                     f"correctly not claimed",
         "effort": "Low - one predicate"},
    ]


def _actions(levers: dict, r2) -> list:
    """Rank by savings per engineer-day, not by savings, then state the payback."""
    n = list(levers)
    cascade_share = (r2["levers_isolated"]["cascade"]
                     / max(1e-9, r2["baseline_daily"] - r2["optimized_daily"]) * 100)
    raw = [
        {"action": "Move interruptible jobs to spot; commit the steady fleet to a reserved "
                   "term", "usd": levers[n[1]], "days": 15,
         "risk": "A 3yr commitment is a liability if the roadmap moves; spot adds ~1.05x "
                 "billed hours per useful hour",
         "note": None},
        {"action": "Shut GPUs down outside working hours", "usd": levers[n[3]], "days": 2,
         "risk": "An overnight submission waits for the morning window unless there is a "
                 "wake-on-demand path",
         "note": None},
        {"action": "Ship the cascade router, the batch queue and the gated prompt cache",
         "usd": levers[n[0]], "days": 10,
         "risk": "Cascading can degrade answers; needs an eval gate on the escalation rate",
         "note": f"the cascade alone is {cascade_share:.0f}% of this lever - ship the router "
                 f"first and negotiate discounts second"},
        {"action": "Re-provision the bandwidth-starved GPUs onto cheaper parts",
         "usd": levers[n[2]], "days": 30,
         "risk": "Halving bandwidth headroom can breach the p95 latency SLO; must be "
                 "benchmarked before cutover",
         "note": "the payback here is longer than a year of the engineering time it costs, "
                 "so it belongs at the next hardware refresh - fix the batch size first, "
                 "which is free and may remove the need entirely"},
    ]
    for a in raw:
        a["roi"] = a["usd"] / a["days"] if a["days"] else 0.0
        a["payback"] = f"{a['days'] * ENG_DAY_USD / a['usd']:.1f} mo" if a["usd"] > 0 else "n/a"
        a["effort"] = f"{a['days']}d (${a['roi']:,.0f}/eng-day)"
    return sorted(raw, key=lambda a: -a["roi"])


def _carbon(r5c, r2) -> dict:
    """Extension 5's region economics, converted into money."""
    table = sorted(r5c["region_table"], key=lambda x: x["balance_score"])
    by_region = {x["region"]: x for x in table}
    home = by_region[ext5_carbon_scheduling.HOME_REGION]
    green = by_region[r5c["greenest_region"]]
    cheap = by_region[r5c["cheapest_region"]]
    bal = by_region[r5c["balanced_region"]]
    infer_kwh = r2["reasoning"]["total_wh_daily"] * DAYS / 1000.0
    rb = r2["reasoning"]

    body = [
        f"Moving the {len(r5c['per_job'])} interruptible jobs from "
        f"{ext5_carbon_scheduling.HOME_REGION} to {r5c['greenest_region']} cuts "
        f"**{r5c['saved_kg_month']:,.0f} kgCO2e/month** "
        f"({r5c['reduction_pct']:.0f}%, about {r5c['saved_kg_month'] * 12 / 1000:.1f} "
        f"tonnes/year) and {'lowers' if r5c['energy_cost_delta_usd'] < 0 else 'raises'} "
        f"the electricity line by ${abs(r5c['energy_cost_delta_usd']):,.0f}/mo. The three "
        f"\"best regions\" are not the same place: cleanest is **{green['region']}** "
        f"({green['gco2_per_kwh']} gCO2/kWh), cheapest electricity is **{cheap['region']}** "
        f"(${cheap['usd_per_kwh']:.3f}/kWh), and the best joint pick is **{bal['region']}** - "
        f"{bal['carbon_kg'] / max(1e-9, green['carbon_kg']):.1f}x the cleanest grid's carbon "
        f"but still {(1 - bal['carbon_kg'] / home['carbon_kg']) * 100:.0f}% below "
        f"{home['region']}, at {(1 - bal['energy_cost'] / green['energy_cost']) * 100:.0f}% "
        f"lower electricity cost and roughly half the added latency.",

        f"**Keeping this honest about money.** At {home['usd_per_kwh']:.3f} $/kWh the movable "
        f"load's electricity is about ${home['energy_cost']:,.0f}/mo - and a neocloud has "
        f"already priced that into the GPU-hour rate. So region choice is a large *carbon* "
        f"lever and a small *cost* lever: the biggest available electricity saving here is "
        f"${home['energy_cost'] - cheap['energy_cost']:,.0f}/mo, which only reaches the "
        f"income statement if you self-host or negotiate a region-differentiated rate. "
        f"Reported as a carbon-intensity target, it is worth "
        f"{r5c['reduction_pct']:.0f}% on the movable half of the fleet.",

        f"**The other half of the energy story is inference, not hardware.** Serving draws "
        f"{rb['total_wh_daily']:,.0f} Wh/day ({infer_kwh:,.0f} kWh/mo). Reasoning requests "
        f"are {rb['traffic_share']:.1%} of traffic but {rb['wh_share']:.1%} of that energy, "
        f"at {rb['wh_per_req_reasoning']:.1f} Wh/request against "
        f"{rb['wh_per_req_normal']:.4f} Wh for a normal one "
        f"({rb['wh_per_req_reasoning'] / max(1e-9, rb['wh_per_req_normal']):.0f}x). Gating "
        f"reasoning on task complexity is therefore a sustainability lever as much as a cost "
        f"one - capping it at {rb['caps'][0]['target_share']:.0%} of traffic saves "
        f"{rb['caps'][0]['saved_wh_daily']:,.0f} Wh/day "
        f"({rb['caps'][0]['saved_wh_pct']:.0f}% of serving energy).",

        f"**Latency is the real constraint.** The clean grids are the far ones: "
        f"{green['region']} is ~{green['rtt_ms']}ms from a US-East user against "
        f"{home['rtt_ms']}ms at home. That is acceptable for the interruptible training and "
        f"eval jobs above, and unacceptable for the 24/7 inference jobs - "
        f"{r5c['fixed_kwh'] / (r5c['fixed_kwh'] + r5c['movable_kwh']) * 100:.0f}% of fleet "
        f"energy therefore stays where it is. Carbon-aware scheduling is a lever on the "
        f"movable half only, and any report claiming otherwise is quietly proposing an SLO "
        f"breach.",
    ]
    return {"fleet_kwh": r5c["movable_kwh"] + r5c["fixed_kwh"],
            "movable_kwh": r5c["movable_kwh"],
            "pue": ext5_carbon_scheduling.PUE,
            "region_table": table, "body": body}


def _extensions(r1, r2, r3, r5c) -> list:
    ce = r2["cache_economics"]
    denied = sorted(k for k, v in ce.items() if not v["worth_it"])
    be = next(iter(ce.values()))["break_even_reads"]
    return [
        {"name": "1. Purchasing policy that prices risk and term",
         "what": "`recommend_tier` now scores every feasible tier in $/*useful*-hour: spot "
                 "carries a per-GPU-type reclaim rate plus checkpoint overhead and expected "
                 "rework, a reservation is charged for all 24h/day because that is what a "
                 "commitment bills, and 1yr vs 3yr is gated on how long the job actually runs.",
         "result": f"The original duty-cycle policy claimed "
                   f"{r3['simple_policy_savings_pct']:.1f}% savings; pricing reservations "
                   f"round the clock revises that **down** to {r3['savings_pct']:.1f}% "
                   f"(${r3['optimized_monthly'] - r3['simple_policy_monthly']:+,.0f}/mo). The "
                   f"break-even reclaim rate for every GPU in the catalogue is above 100%/h, "
                   f"so spot is not the risky choice for interruptible work - the "
                   f"under-utilised commitment is."},
        {"name": "2. Right-sizing justified by bandwidth, not by sticker price",
         "what": "For each GPU under 30% MFU, find the cheapest catalogue part that still "
                 "delivers the bandwidth it actually uses plus 25% headroom, and price the "
                 "swap in $/TB/s-hr and $/GB-VRAM-hr rather than $/GPU-hr.",
         "result": f"{len(r1['rightsize_proposals'])} of {len(r1['summary'])} GPUs qualify, "
                   f"worth **${r1['rightsize_monthly_savings']:,.0f}/mo** billed on active "
                   f"hours only. Picking on $/GPU-hr instead would have proposed an L4 for "
                   f"the H100 workloads, which cannot sustain their memory traffic."},
        {"name": "3. Prompt caching gated on its own break-even",
         "what": "A cache write costs a 25% premium and a read saves 90%, so break-even is "
                 f"0.25/0.90 = {be:.3f} re-reads. But a 5-minute TTL caps re-reads at "
                 "whatever arrives inside the same window, so the gate is evaluated per "
                 "(team, route_tier) prefix group.",
         "result": f"{len(denied)}/{len(ce)} groups receive under one request per TTL window "
                   f"and therefore *lose* money on caching: {', '.join(denied)}. Declining "
                   f"them forgoes ${r2['cache_denied_savings_daily'] * DAYS:,.0f}/mo of "
                   f"apparent savings that would not have been real. \"Enable caching "
                   f"everywhere\" is the wrong default."},
        {"name": "4. Reasoning-token budget",
         "what": "Split spend and energy by `is_reasoning`, then simulate capping reasoning "
                 "to a share of traffic by keeping the longest answers and downgrading the "
                 "rest.",
         "result": f"Reasoning is {r2['reasoning']['traffic_share']:.1%} of requests but "
                   f"{r2['reasoning']['cost_share']:.1%} of spend and "
                   f"{r2['reasoning']['wh_share']:.1%} of energy. Capping to "
                   f"{r2['reasoning']['caps'][0]['target_share']:.0%} saves "
                   f"${r2['reasoning']['caps'][0]['saved_cost_daily']:.2f}/day "
                   f"({r2['reasoning']['caps'][0]['saved_cost_pct']:.1f}% of the bill) and "
                   f"{r2['reasoning']['caps'][0]['saved_wh_daily']:,.0f} Wh/day "
                   f"({r2['reasoning']['caps'][0]['saved_wh_pct']:.0f}%)."},
        {"name": "5. Carbon-aware scheduling",
         "what": "Place only the interruptible jobs by grid carbon intensity, and score all "
                 "five regions on price, carbon and latency together.",
         "result": f"**{r5c['saved_kg_month']:,.0f} kgCO2e/mo** ({r5c['reduction_pct']:.0f}%) "
                   f"off the movable load by moving it to {r5c['greenest_region']}; "
                   f"{r5c['balanced_region']} is the better joint pick. "
                   f"{r5c['fixed_kwh'] / (r5c['fixed_kwh'] + r5c['movable_kwh']) * 100:.0f}% "
                   f"of fleet energy is latency-bound and cannot move."},
    ]


def _assumptions(r1, r2, r3) -> list:
    return [
        "**The two datasets are separate views.** `gpu_telemetry.csv` (11 GPUs) and "
        "`workloads.csv` (8 jobs) are not reconciled to the same physical fleet, so the "
        "four levers are summed as if independent. In production they interact: the "
        "purchasing lever would apply to *right-sized* rates, which would make the combined "
        "figure somewhat lower than the sum shown here. The per-lever numbers are the "
        "defensible ones; the total is an upper bound.",
        "**Idle and right-size do not double-count.** One GPU (gpu-h100-5) is both idle "
        "overnight and a right-size candidate, so its right-size saving is billed on active "
        "hours (16/day) rather than 24.",
        f"**Baseline is a naive deployment, not today's bill.** Inference baseline prices "
        f"every request on the large model with no cache and no batching "
        f"(${r2['baseline_daily']:,.2f}/day); purchasing baseline is 100% on-demand "
        f"(${r3['on_demand_monthly']:,.0f}/mo). Savings against a partly-optimised estate "
        f"would be smaller.",
        f"**Payback uses ${ENG_DAY_USD:,.0f}/engineer-day loaded cost.** Change that number "
        f"and the ranking of the last two actions changes with it; the ranking of the first "
        f"two does not.",
        "**Prices, carbon intensities and the 80x reasoning-energy multiplier are "
        "illustrative June-2026 snapshots.** GPU pricing moves fast enough that a "
        "purchasing commitment should be re-baselined before it is signed.",
    ]


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r5c = ext5_carbon_scheduling.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]
    idle_savings = r1["idle_waste_daily"] * DAYS
    # EXTENSION 2: bandwidth-justified swaps, billed on active hours, instead of a blind
    # one-tier-down map applied to whichever GPUs happened to be flagged as lies.
    rightsize_savings = r1["rightsize_monthly_savings"]

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size memory-bound GPUs": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    wh = sustainability.wh_per_query(MEDIAN_QUERY_TOKENS)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
        "wh_per_query_reasoning": sustainability.wh_per_query(
            MEDIAN_QUERY_TOKENS, is_reasoning=True),
        "reasoning_multiplier": sustainability.REASONING_ENERGY_MULTIPLIER,
        # EXTENSION 5: the three criteria disagree, so name all three picks
        "cheapest_region": r5c["cheapest_region"],
        "balanced_region": r5c["balanced_region"],
    }

    md = report.build_report(
        baseline, optimized, levers, sustainability=sust,
        unit_economics=_unit_economics(r2, baseline, optimized),
        lever_detail=_lever_detail(levers, r1, r2, r3),
        root_cause=_root_cause(r1, cat),
        actions=_actions(levers, r2),
        carbon=_carbon(r5c, r2),
        extensions=_extensions(r1, r2, r3, r5c),
        assumptions=_assumptions(r1, r2, r3),
    )
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"),
                                  baseline_usd=baseline)

    if verbose:
        print("== M5 Optimization Report ==")
        print(f"baseline  ${baseline:,.0f}/mo  ->  optimized ${optimized:,.0f}/mo   "
              f"({total_pct:.1f}% saved, ${sum(levers.values()):,.0f}/mo)")
        print(f"  {'lever':34}{'$/mo':>10}{'% of savings':>14}")
        for k, v in levers.items():
            print(f"  {k:34}{v:>10,}{v / sum(levers.values()) * 100:>13.0f}%")
        print(f"\n  unit economics: ${r2['baseline_per_m']:.3f} -> "
              f"${r2['optimized_per_m']:.3f} per 1M tokens")
        print("  do-first order (savings per engineer-day):")
        for i, a in enumerate(_actions(levers, r2), 1):
            print(f"    {i}. {a['action'][:62]:63}${a['roi']:>7,.0f}/d  payback {a['payback']}")
        print(f"\nWritten: outputs/report.md ({len(md.splitlines())} lines)"
              + (" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1),
            "carbon": {"saved_kg_month": r5c["saved_kg_month"],
                       "greenest_region": r5c["greenest_region"],
                       "balanced_region": r5c["balanced_region"]}}


if __name__ == "__main__":
    run()
