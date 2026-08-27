"""EXTENSION 5 — Carbon-aware scheduling for interruptible GPU jobs (deck §11).

Run: python missions/ext5_carbon_scheduling.py

Interruptible jobs are the ones that can be MOVED: nothing about a checkpointable
training run requires it to sit next to the users. That makes them the natural
candidates for placing on a clean grid. Jobs serving live traffic cannot move
without paying a latency penalty, which is the real constraint here.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

DAYS = 30
HOME_REGION = "us-east-1"      # where NimbusAI runs everything today
PUE = 1.12                     # datacenter overhead: cooling, power conversion, network

# Illustrative round-trip latency from a US-East user population (ms). The clean
# grids are the far ones — that is the whole trade-off.
REGION_RTT_MS = {
    "us-east-1": 15,
    "us-east-wa": 60,
    "us-west-2": 70,
    "europe-north1": 110,
    "europe-central2": 120,
}


def job_energy_kwh(job, cat, days: int = DAYS) -> float:
    """Monthly grid energy drawn by a job, including datacenter overhead (PUE)."""
    watts = num(cat[job["gpu_type"]]["watts"])
    gpu_hours = num(job["hours_per_day"]) * days * int(num(job["num_gpus"]))
    return watts * gpu_hours / 1000.0 * PUE


def region_table(total_kwh: float) -> list:
    """Price/carbon/latency for every region at a given energy volume."""
    regions = sorted(sustainability.REGION_CARBON)
    rows = []
    for r in regions:
        rows.append({
            "region": r,
            "gco2_per_kwh": sustainability.REGION_CARBON[r],
            "usd_per_kwh": sustainability.REGION_PRICE_KWH.get(r, 0.12),
            "energy_cost": sustainability.energy_cost_usd(total_kwh * 1000.0, r),
            "carbon_kg": sustainability.carbon_g(total_kwh * 1000.0, r) / 1000.0,
            "rtt_ms": REGION_RTT_MS.get(r, 0),
        })
    # "Balanced" = min-max normalise cost and carbon, then take the lowest sum. Both
    # dimensions get equal weight, so neither a dirty-but-cheap nor a clean-but-dear
    # region can win on one axis alone.
    costs = [x["energy_cost"] for x in rows]
    carbons = [x["carbon_kg"] for x in rows]
    for x in rows:
        nc = ((x["energy_cost"] - min(costs)) / (max(costs) - min(costs))
              if max(costs) > min(costs) else 0.0)
        ng = ((x["carbon_kg"] - min(carbons)) / (max(carbons) - min(carbons))
              if max(carbons) > min(carbons) else 0.0)
        x["balance_score"] = round(nc + ng, 3)
    return rows


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()

    movable = [j for j in jobs if bool(int(num(j["interruptible"])))]
    fixed = [j for j in jobs if not bool(int(num(j["interruptible"])))]

    cleanest = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    movable_kwh = sum(job_energy_kwh(j, cat) for j in movable)
    fixed_kwh = sum(job_energy_kwh(j, cat) for j in fixed)

    per_job = []
    for j in movable:
        kwh = job_energy_kwh(j, cat)
        wh = kwh * 1000.0
        home_c = sustainability.carbon_g(wh, HOME_REGION)
        best_c = sustainability.carbon_g(wh, cleanest)
        per_job.append({
            "job_id": j["job_id"], "gpu_type": j["gpu_type"],
            "num_gpus": int(num(j["num_gpus"])), "kwh_month": round(kwh, 1),
            "home_kg": round(home_c / 1000.0, 1), "best_kg": round(best_c / 1000.0, 1),
            "saved_kg": round((home_c - best_c) / 1000.0, 1),
            "home_energy_usd": round(sustainability.energy_cost_usd(wh, HOME_REGION), 2),
            "best_energy_usd": round(sustainability.energy_cost_usd(wh, cleanest), 2),
        })

    saved_kg = sum(p["saved_kg"] for p in per_job)
    home_kg = sum(p["home_kg"] for p in per_job)
    reduction_pct = saved_kg / home_kg * 100 if home_kg else 0.0
    saved_usd = sum(p["home_energy_usd"] - p["best_energy_usd"] for p in per_job)

    table = region_table(movable_kwh)
    cheapest = min(table, key=lambda x: x["energy_cost"])
    greenest = min(table, key=lambda x: x["carbon_kg"])
    balanced = min(table, key=lambda x: x["balance_score"])

    if verbose:
        print("== EXTENSION 5: Carbon-aware scheduling ==")
        print(f"home region = {HOME_REGION}   PUE = {PUE}   horizon = {DAYS} days")
        print(f"movable (interruptible) load: {len(movable)} jobs, {movable_kwh:,.0f} kWh/mo")
        print(f"fixed (latency-bound) load  : {len(fixed)} jobs, {fixed_kwh:,.0f} kWh/mo "
              f"({fixed_kwh/(movable_kwh+fixed_kwh)*100:.0f}% of energy cannot move)")

        print(f"\nper-job: {HOME_REGION} vs cleanest grid ({cleanest})")
        print(f"  {'job':18}{'gpu':7}{'kWh/mo':>9}{'kgCO2 home':>12}{'kgCO2 best':>12}{'saved':>9}{'% cut':>8}")
        for p in per_job:
            cut = p["saved_kg"] / p["home_kg"] * 100 if p["home_kg"] else 0
            print(f"  {p['job_id']:18}{p['gpu_type']:7}{p['kwh_month']:>9,.0f}{p['home_kg']:>12,.1f}"
                  f"{p['best_kg']:>12,.1f}{p['saved_kg']:>9,.1f}{cut:>7.0f}%")
        print(f"  {'TOTAL':18}{'':7}{movable_kwh:>9,.0f}{home_kg:>12,.1f}"
              f"{home_kg-saved_kg:>12,.1f}{saved_kg:>9,.1f}{reduction_pct:>7.0f}%")
        print(f"\n  -> moving all interruptible jobs to {cleanest} cuts {saved_kg:,.0f} kgCO2e/month "
              f"({reduction_pct:.0f}%),")
        print(f"     about {saved_kg*12/1000:,.1f} tonnes/year, and changes the electricity bill by "
              f"${-saved_usd:+,.2f}/mo.")

        print(f"\nall regions, priced on the movable {movable_kwh:,.0f} kWh/mo:")
        print(f"  {'region':16}{'$/kWh':>8}{'gCO2/kWh':>10}{'energy $':>11}{'tCO2e':>9}{'RTT':>7}{'score':>8}")
        for x in sorted(table, key=lambda y: y["balance_score"]):
            print(f"  {x['region']:16}{x['usd_per_kwh']:>8.3f}{x['gco2_per_kwh']:>10}"
                  f"{x['energy_cost']:>11,.0f}{x['carbon_kg']/1000:>9.2f}{x['rtt_ms']:>6}ms"
                  f"{x['balance_score']:>8.2f}")
        print(f"\n  cheapest electricity : {cheapest['region']:16} ${cheapest['energy_cost']:,.0f}/mo, "
              f"{cheapest['carbon_kg']/1000:.2f} tCO2e")
        print(f"  cleanest grid        : {greenest['region']:16} ${greenest['energy_cost']:,.0f}/mo, "
              f"{greenest['carbon_kg']/1000:.2f} tCO2e")
        print(f"  best balance         : {balanced['region']:16} ${balanced['energy_cost']:,.0f}/mo, "
              f"{balanced['carbon_kg']/1000:.2f} tCO2e")
        print("\n  WHICH IS 'OPTIMAL' DEPENDS ON THE COMPANY'S PRIORITY:")
        print(f"  - carbon-first  -> {greenest['region']} (hydro, {greenest['gco2_per_kwh']} gCO2/kWh) but "
              f"{REGION_RTT_MS[greenest['region']]}ms from US users")
        print(f"  - cost-first    -> {cheapest['region']} (${cheapest['usd_per_kwh']}/kWh) and still only "
              f"{cheapest['gco2_per_kwh']} gCO2/kWh")
        home_row = next(x for x in table if x["region"] == HOME_REGION)
        print(f"  - balanced      -> {balanced['region']}: "
              f"{balanced['carbon_kg']/max(1e-9,greenest['carbon_kg']):.1f}x the cleanest grid's carbon, "
              f"but still {(1-balanced['carbon_kg']/max(1e-9,home_row['carbon_kg']))*100:.0f}% below "
              f"{HOME_REGION}, at "
              f"{(1-balanced['energy_cost']/max(1e-9,greenest['energy_cost']))*100:.0f}% lower "
              "electricity cost and half the added latency")
        print("  LATENCY TRADE-OFF: the clean grids are the far ones. That is fine for the")
        print("  interruptible training/eval jobs priced above (nobody waits on them), and wrong")
        print("  for the 24/7 inference jobs, which is exactly why only the movable half moves.")
        print(f"  Note: at {home_row['usd_per_kwh']:.3f} $/kWh the electricity is ~${home_row['energy_cost']:,.0f}/mo "
              "of a GPU-hour bill that already embeds it,")
        print("  so region choice mostly moves CARBON; the dollars only follow if you self-host.")

    return {"cleanest_region": cleanest, "movable_kwh": round(movable_kwh, 1),
            "fixed_kwh": round(fixed_kwh, 1), "per_job": per_job,
            "saved_kg_month": round(saved_kg, 1), "home_kg_month": round(home_kg, 1),
            "reduction_pct": round(reduction_pct, 1),
            "energy_cost_delta_usd": round(-saved_usd, 2),
            "region_table": table, "cheapest_region": cheapest["region"],
            "greenest_region": greenest["region"], "balanced_region": balanced["region"]}


if __name__ == "__main__":
    run()
