"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py

EXTENSION 2 is wired in here: right-sizing by MBU. A memory-bound GPU is chosen
for its HBM bandwidth, not its FLOPs, so the right question is "$ per TB/s and per
GB of VRAM", not "$ per GPU-hour".
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

DAYS = 30
# A GPU whose MFU is low but whose MBU is comparatively high is memory-bound: it is
# starved on bandwidth, not on compute, so paying for peak FLOPs is waste.
MEMORY_BOUND_MFU_MAX = 0.30


def unit_economics(cat: dict) -> dict:
    """EXTENSION 2 — cost per unit of the resource that actually limits the workload."""
    out = {}
    for gtype, c in cat.items():
        od = num(c["on_demand_hr"])
        hbm = num(c["hbm_gb"])
        bw = num(c["peak_bw_tbs"])
        tflops = num(c["peak_tflops_fp16"])
        out[gtype] = {
            "on_demand_hr": od,
            "hbm_gb": hbm,
            "peak_bw_tbs": bw,
            "peak_tflops": tflops,
            "usd_per_gb_vram_hr": od / hbm if hbm else float("inf"),
            "usd_per_tbs_hr": od / bw if bw else float("inf"),
            "usd_per_tflop_hr": od / tflops if tflops else float("inf"),
        }
    return out


def rightsize_by_mbu(summary, cat: dict) -> list:
    """For each memory-bound GPU, find the cheapest part that still serves its
    achieved bandwidth, and price the swap.

    The candidate must deliver at least the bandwidth the GPU is *actually* using
    (achieved BW, with headroom) — replacing on sticker price alone would move the
    workload onto a part that cannot sustain its memory traffic.

    Savings are billed on the hours the GPU is ACTUALLY UP (24 - idle), so a GPU that
    also appears in the "kill idle" lever is not counted twice.
    """
    econ = unit_economics(cat)
    HEADROOM = 1.25          # keep 25% bandwidth headroom over what we observe today
    proposals = []
    for s in summary:
        cur = s["gpu_type"]
        if s["mfu"] >= MEMORY_BOUND_MFU_MAX:
            continue                                  # compute-bound enough to keep
        cur_c = econ[cur]
        achieved_bw = s["mbu"] * cur_c["peak_bw_tbs"]
        need_bw = achieved_bw * HEADROOM
        need_vram = num(cat[cur]["hbm_gb"]) * 0.5     # assume model fits in half of today's VRAM
        # cheapest catalog part that still clears the bandwidth + VRAM floor
        feasible = [(g, e) for g, e in econ.items()
                    if e["peak_bw_tbs"] >= need_bw and e["hbm_gb"] >= need_vram]
        if not feasible:
            continue
        best, best_e = min(feasible, key=lambda kv: kv[1]["on_demand_hr"])
        if best_e["on_demand_hr"] >= cur_c["on_demand_hr"]:
            continue                                  # already on the cheapest fit
        delta_hr = cur_c["on_demand_hr"] - best_e["on_demand_hr"]
        # only the hours this GPU is genuinely up: the idle hours are already being
        # claimed by the "kill idle GPUs" lever, and claiming them twice inflates the
        # total savings by exactly the overlap.
        active_hours = max(0.0, 24.0 - s["idle_hours"])
        proposals.append({
            "gpu_id": s["gpu_id"], "current": cur, "proposed": best,
            "mfu": s["mfu"], "mbu": s["mbu"],
            "achieved_bw_tbs": round(achieved_bw, 3),
            "need_bw_tbs": round(need_bw, 3),
            "cur_hr": cur_c["on_demand_hr"], "new_hr": best_e["on_demand_hr"],
            "cur_usd_per_tbs": round(cur_c["usd_per_tbs_hr"], 3),
            "new_usd_per_tbs": round(best_e["usd_per_tbs_hr"], 3),
            "cur_usd_per_gb": round(cur_c["usd_per_gb_vram_hr"], 4),
            "new_usd_per_gb": round(best_e["usd_per_gb_vram_hr"], 4),
            "savings_pct": round(delta_hr / cur_c["on_demand_hr"] * 100, 1),
            "active_hours": active_hours,
            "monthly_savings": round(delta_hr * active_hours * DAYS, 2),
        })
    return proposals


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(cat[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- EXTENSION 2: right-size the memory-bound GPUs ---
    econ = unit_economics(cat)
    proposals = rightsize_by_mbu(summary, cat)
    rightsize_monthly = sum(p["monthly_savings"] for p in proposals)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*DAYS:,.0f}/month")

        print("\n-- EXTENSION 2: right-sizing memory-bound GPUs by MBU --")
        print("  catalog unit economics (why $/GPU-hr is the wrong yardstick):")
        print(f"    {'gpu':8}{'$/hr':>7}{'VRAM':>6}{'TB/s':>7}{'$/GB-hr':>10}{'$/TB/s-hr':>11}{'$/TFLOP-hr':>12}")
        for g, e in sorted(econ.items(), key=lambda kv: kv[1]["usd_per_tbs_hr"]):
            print(f"    {g:8}{e['on_demand_hr']:>7.2f}{e['hbm_gb']:>6.0f}{e['peak_bw_tbs']:>7.2f}"
                  f"{e['usd_per_gb_vram_hr']:>10.4f}{e['usd_per_tbs_hr']:>11.3f}{e['usd_per_tflop_hr']:>12.5f}")
        if proposals:
            print(f"\n  swaps for GPUs with MFU < {MEMORY_BOUND_MFU_MAX:.0%} (bandwidth-starved, not compute-starved):")
            print(f"    {'GPU':14}{'now':6}{'->':3}{'proposed':10}{'MFU':>6}{'MBU':>6}"
                  f"{'BW used':>9}{'BW need':>9}{'$/TB/s now':>11}{'->new':>7}{'up h/d':>8}{'save/mo':>10}")
            for p in proposals:
                print(f"    {p['gpu_id']:14}{p['current']:6}{'->':3}{p['proposed']:10}"
                      f"{p['mfu']:>6.2f}{p['mbu']:>6.2f}{p['achieved_bw_tbs']:>9.2f}{p['need_bw_tbs']:>9.2f}"
                      f"{p['cur_usd_per_tbs']:>11.3f}{p['new_usd_per_tbs']:>7.3f}{p['active_hours']:>8.0f}"
                      f"{'$' + format(p['monthly_savings'], ',.0f'):>10}")
            print(f"  right-size savings if all {len(proposals)} are moved: ${rightsize_monthly:,.0f}/month")
            print("  NOTE: the pick is the cheapest part that still clears the bandwidth the GPU")
            print("  actually uses (+25% headroom) - NOT the cheapest $/GPU-hr, which would")
            print("  starve a memory-bound workload and make latency (and cost/token) worse.")
            print("  Savings are billed on 'up h/d' (24 - idle), so a GPU that also shows up in")
            print("  the idle-waste lever is not double-counted.")
        else:
            print("  no memory-bound GPU has a cheaper part that still clears its bandwidth floor.")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "unit_economics": econ, "rightsize_proposals": proposals,
            "rightsize_monthly_savings": round(rightsize_monthly, 2)}


if __name__ == "__main__":
    run()
