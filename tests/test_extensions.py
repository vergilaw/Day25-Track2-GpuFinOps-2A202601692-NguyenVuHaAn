"""Tests for the five extensions. This is a NEW test file: the provided tests in
tests/ are left untouched.

Each test states a property the extension must satisfy, derived from the economics
rather than from a recorded output, so none of them can pass by hardcoding.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finops import pricing, sustainability
from missions import (ext5_carbon_scheduling, m1_efficiency_audit,
                      m2_inference_levers, m3_purchasing)
from missions._common import catalog_by_type, num

H100 = {"on_demand_hr": 2.5, "spot_hr": 1.5, "reserved_1yr_hr": 2.0, "reserved_3yr_hr": 1.4}


# ---------------------------------------------------------------------------
# EXTENSION 1 - purchasing policy that prices reclaim risk and commitment term
# ---------------------------------------------------------------------------
def test_documented_simple_policy_is_unchanged():
    """Calling recommend_tier without prices must keep the original behaviour."""
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_spot_multiplier_always_exceeds_one_and_rises_with_risk():
    """A spot hour never buys a full hour of useful work, and a churnier pool costs more."""
    calm = pricing.spot_effective_multiplier(0.03)
    churny = pricing.spot_effective_multiplier(0.12)
    assert calm > 1.0
    assert churny > calm
    assert pricing.spot_effective_multiplier(0.0) == 1.03


def test_interrupt_rate_is_gpu_specific():
    """Commodity inference parts churn more than scarce training parts."""
    assert pricing.spot_interrupt_rate("A10G") > pricing.spot_interrupt_rate("H100")
    assert pricing.spot_interrupt_rate("nonexistent") == pricing.DEFAULT_INTERRUPT_RATE


def test_break_even_interrupt_rate_inverts_the_multiplier():
    """At the break-even rate, spot and on-demand cost the same per useful hour."""
    be = pricing.break_even_interrupt_rate(H100["spot_hr"], H100["on_demand_hr"])
    cost_at_be = H100["spot_hr"] * pricing.spot_effective_multiplier(be)
    assert abs(cost_at_be - H100["on_demand_hr"]) < 1e-9
    # just below break-even spot wins, just above it loses
    assert H100["spot_hr"] * pricing.spot_effective_multiplier(be * 0.9) < H100["on_demand_hr"]
    assert H100["spot_hr"] * pricing.spot_effective_multiplier(be * 1.1) > H100["on_demand_hr"]


def test_reservation_is_billed_round_the_clock():
    """The core correction: a commitment bills 24h/day regardless of hours used."""
    full = pricing.effective_hourly_cost("reserved", 24, H100)
    half = pricing.effective_hourly_cost("reserved", 12, H100)
    assert abs(full - H100["reserved_3yr_hr"]) < 1e-9
    assert abs(half - H100["reserved_3yr_hr"] * 2) < 1e-9      # half the hours, twice the rate
    # on-demand does not have this property
    assert (pricing.effective_hourly_cost("on_demand", 24, H100)
            == pricing.effective_hourly_cost("on_demand", 3, H100))


def test_reservation_beats_on_demand_exactly_at_the_break_even_utilization():
    """break_even_utilization(d) = 1-d must be the crossover point, not a rule of thumb."""
    discount = 1.0 - H100["reserved_3yr_hr"] / H100["on_demand_hr"]
    be_hours = pricing.break_even_utilization(discount) * 24.0
    at = pricing.effective_hourly_cost("reserved", be_hours, H100)
    assert abs(at - H100["on_demand_hr"]) < 1e-6
    assert pricing.effective_hourly_cost("reserved", be_hours + 1, H100) < H100["on_demand_hr"]
    assert pricing.effective_hourly_cost("reserved", be_hours - 1, H100) > H100["on_demand_hr"]


def test_commit_term_follows_job_duration():
    assert pricing.commit_term(30) == "3yr"
    assert pricing.commit_term(10) == "1yr"
    assert pricing.commit_term(2) is None                       # too short to commit
    assert pricing.commit_term(None) == "3yr"                    # legacy steady-state default


def test_short_non_interruptible_job_falls_back_to_on_demand():
    """No feasible commitment and no spot eligibility leaves only on-demand."""
    d = pricing.recommend_tier_detailed(24, False, gpu_type="H100", job_days=2, prices=H100)
    assert d["tier"] == "on_demand"
    assert "reserved" not in d["candidates"]
    assert d["term"] is None


def test_detailed_policy_picks_the_cheapest_feasible_tier():
    """Whatever the rationale text says, the chosen tier must be the cost-minimising one."""
    for hpd in (1, 4, 8, 12, 18, 24):
        for interruptible in (True, False):
            for days in (2, 10, 30):
                d = pricing.recommend_tier_detailed(
                    hpd, interruptible, gpu_type="H100", job_days=days, prices=H100)
                assert d["tier"] == min(d["candidates"], key=d["candidates"].get)
                assert d["effective_hr"] == min(d["candidates"].values())
                if not interruptible:
                    assert "spot" not in d["candidates"]


def test_m3_economic_policy_is_not_more_optimistic_than_the_simple_one():
    """Pricing reservations 24/7 can only remove savings the naive model invented."""
    r3 = m3_purchasing.run(verbose=False)
    assert r3["optimized_monthly"] >= r3["simple_policy_monthly"]
    assert r3["savings_pct"] <= r3["simple_policy_savings_pct"]
    assert 0 < r3["savings_pct"] < 100


# ---------------------------------------------------------------------------
# EXTENSION 2 - right-sizing justified by bandwidth, not by sticker price
# ---------------------------------------------------------------------------
def test_unit_economics_are_per_limiting_resource():
    econ = m1_efficiency_audit.unit_economics(catalog_by_type())
    h100, l4 = econ["H100"], econ["L4"]
    assert h100["usd_per_tbs_hr"] < l4["usd_per_tbs_hr"]     # H100 is cheaper per TB/s...
    assert h100["on_demand_hr"] > l4["on_demand_hr"]         # ...while costing more per hour
    for e in econ.values():
        assert abs(e["usd_per_gb_vram_hr"] - e["on_demand_hr"] / e["hbm_gb"]) < 1e-9


def test_rightsize_proposals_never_starve_the_workload():
    """Every proposal must clear the bandwidth the GPU actually uses, plus headroom."""
    r1 = m1_efficiency_audit.run(verbose=False)
    cat = catalog_by_type()
    assert r1["rightsize_proposals"], "expected at least one memory-bound GPU in the fixture"
    for p in r1["rightsize_proposals"]:
        assert p["mfu"] < m1_efficiency_audit.MEMORY_BOUND_MFU_MAX
        assert num(cat[p["proposed"]]["peak_bw_tbs"]) >= p["need_bw_tbs"]
        assert p["need_bw_tbs"] > p["achieved_bw_tbs"]        # headroom is real
        assert p["new_hr"] < p["cur_hr"]                      # and it is cheaper
        assert p["monthly_savings"] > 0


def test_rightsize_does_not_double_count_idle_hours():
    """A GPU claimed by the idle lever cannot also be right-sized for those hours."""
    r1 = m1_efficiency_audit.run(verbose=False)
    idle = {s["gpu_id"]: s["idle_hours"] for s in r1["summary"]}
    for p in r1["rightsize_proposals"]:
        assert p["active_hours"] == 24 - idle[p["gpu_id"]]
        expected = (p["cur_hr"] - p["new_hr"]) * p["active_hours"] * m1_efficiency_audit.DAYS
        assert abs(p["monthly_savings"] - expected) < 0.01
    assert any(idle[p["gpu_id"]] > 0 for p in r1["rightsize_proposals"]), \
        "fixture should contain a GPU that is both idle and right-sizable"


# ---------------------------------------------------------------------------
# EXTENSION 3 - prompt caching gated on its own break-even
# ---------------------------------------------------------------------------
def test_break_even_cache_reads_is_the_write_over_read_saving_ratio():
    be = pricing.break_even_cache_reads(0.25, 0.10, 1.0)
    assert abs(be - 0.25 / 0.90) < 1e-9
    # price-independent: scaling both sides leaves the ratio alone
    assert abs(pricing.break_even_cache_reads(3.0 * 0.25, 0.10, 3.0) - be) < 1e-9
    # a bigger write premium needs more re-reads; a deeper read discount needs fewer
    assert pricing.break_even_cache_reads(0.50, 0.10, 1.0) > be
    assert pricing.break_even_cache_reads(0.25, 0.50, 1.0) > be


def test_cache_gate_flips_at_the_break_even():
    be = pricing.break_even_cache_reads(0.25, 0.10, 1.0)
    assert pricing.cache_is_worth_it(be + 0.01, 0.25, 0.10, 1.0)
    assert not pricing.cache_is_worth_it(be - 0.01, 0.25, 0.10, 1.0)
    assert not pricing.cache_is_worth_it(0.0, 0.25, 0.10, 1.0)   # written, never read


def test_ttl_bounds_reads_per_write_and_the_gate_uses_it():
    """The gate must be driven by requests per TTL window, not requests per day."""
    econ = m2_inference_levers.cache_economics(
        m2_inference_levers.load_csv("token_usage.csv"))
    assert econ, "expected prefix groups"
    for e in econ.values():
        expected = e["requests"] / m2_inference_levers.WINDOWS_PER_DAY
        assert abs(e["reqs_per_window"] - round(expected, 2)) < 0.01
        assert e["reads_per_write"] == round(max(0.0, expected - 1.0), 2)
        assert e["worth_it"] == (e["reads_per_write"] > e["break_even_reads"])
        # the naive per-day count would almost always say yes; the TTL view need not
        assert e["naive_reads_per_day"] >= e["reads_per_write"]
    assert any(not e["worth_it"] for e in econ.values()), \
        "a low-traffic group should fail the gate, else the extension proves nothing"


def test_gated_cache_is_never_more_expensive_than_no_cache():
    r2 = m2_inference_levers.run(verbose=False)
    assert r2["cache_denied_savings_daily"] >= 0      # savings foregone, not negative savings
    assert r2["levers"]["caching"] >= 0               # the lever we do claim is a real saving
    assert r2["optimized_daily"] < r2["baseline_daily"]


# ---------------------------------------------------------------------------
# EXTENSION 4 - reasoning-token budget
# ---------------------------------------------------------------------------
def test_reasoning_split_reconciles_with_the_optimized_total():
    """The two halves must add back up to the bill they were split out of."""
    r2 = m2_inference_levers.run(verbose=False)
    rb = r2["reasoning"]
    total = rb["reasoning_cost_daily"] + rb["normal_cost_daily"]
    assert abs(total - r2["optimized_daily"]) < 0.02
    assert rb["reasoning_requests"] + rb["normal_requests"] > 0


def test_reasoning_is_disproportionately_expensive_and_energy_hungry():
    rb = m2_inference_levers.run(verbose=False)["reasoning"]
    assert 0 < rb["traffic_share"] < 1
    assert rb["cost_share"] > rb["traffic_share"]     # costs more than its share of traffic
    assert rb["wh_share"] > rb["cost_share"]          # and far more than its share of spend
    assert rb["wh_per_req_reasoning"] > rb["wh_per_req_normal"]


def test_reasoning_energy_multiplier_is_applied():
    n = sustainability.wh_per_query(1000, is_reasoning=False)
    r = sustainability.wh_per_query(1000, is_reasoning=True)
    assert abs(r / n - sustainability.REASONING_ENERGY_MULTIPLIER) < 1e-9


def test_tighter_reasoning_cap_saves_more():
    """Caps are monotone: allowing less reasoning cannot save less."""
    caps = m2_inference_levers.run(verbose=False)["reasoning"]["caps"]
    ordered = sorted(caps, key=lambda c: -c["target_share"])
    for a, b in zip(ordered, ordered[1:]):
        assert b["saved_cost_daily"] >= a["saved_cost_daily"]
        assert b["saved_wh_daily"] >= a["saved_wh_daily"]
        assert b["downgraded"] >= a["downgraded"]
    for c in caps:
        assert c["kept"] + c["downgraded"] > 0
        assert 0 <= c["saved_wh_pct"] <= 100


# ---------------------------------------------------------------------------
# EXTENSION 5 - carbon-aware scheduling
# ---------------------------------------------------------------------------
def test_only_interruptible_jobs_are_treated_as_movable():
    r5 = ext5_carbon_scheduling.run(verbose=False)
    moved = {p["job_id"] for p in r5["per_job"]}
    for j in ext5_carbon_scheduling.load_csv("workloads.csv"):
        assert (j["job_id"] in moved) == bool(int(num(j["interruptible"])))
    assert r5["fixed_kwh"] > 0, "some load must be latency-bound, else there is no trade-off"


def test_energy_includes_pue_and_scales_with_gpu_count():
    cat = catalog_by_type()
    job = {"gpu_type": "H100", "hours_per_day": "10", "num_gpus": "4"}
    kwh = ext5_carbon_scheduling.job_energy_kwh(job, cat, days=30)
    expected = 700 * 10 * 30 * 4 / 1000.0 * ext5_carbon_scheduling.PUE
    assert abs(kwh - expected) < 1e-6
    one = ext5_carbon_scheduling.job_energy_kwh({**job, "num_gpus": "1"}, cat, days=30)
    assert abs(kwh - 4 * one) < 1e-6


def test_moving_to_the_cleanest_grid_cuts_carbon_by_the_intensity_ratio():
    r5 = ext5_carbon_scheduling.run(verbose=False)
    home = sustainability.REGION_CARBON[ext5_carbon_scheduling.HOME_REGION]
    best = sustainability.REGION_CARBON[r5["cleanest_region"]]
    assert r5["cleanest_region"] == min(sustainability.REGION_CARBON,
                                       key=sustainability.REGION_CARBON.get)
    assert abs(r5["reduction_pct"] - (1 - best / home) * 100) < 0.5
    assert r5["saved_kg_month"] > 0


def test_region_table_is_complete_and_the_three_picks_are_defensible():
    r5 = ext5_carbon_scheduling.run(verbose=False)
    table = {x["region"]: x for x in r5["region_table"]}
    assert set(table) == set(sustainability.REGION_CARBON)
    assert r5["cheapest_region"] == min(table, key=lambda r: table[r]["energy_cost"])
    assert r5["greenest_region"] == min(table, key=lambda r: table[r]["carbon_kg"])
    assert r5["balanced_region"] == min(table, key=lambda r: table[r]["balance_score"])
    # the balanced pick must not be worst on either axis it is balancing
    bal = table[r5["balanced_region"]]
    assert bal["energy_cost"] < max(x["energy_cost"] for x in table.values())
    assert bal["carbon_kg"] < max(x["carbon_kg"] for x in table.values())
    for x in table.values():
        assert x["balance_score"] >= 0
