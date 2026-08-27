"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py

EXTENSION 3 (cache_is_worth_it) and EXTENSION 4 (reasoning budget) are wired in
here. The cache lever is GATED: a cached prefix is only billed as a saving when it
is actually re-read enough times to recoup the cache-write premium.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# EXTENSION 3 — cache economics. Anthropic-style 5-minute TTL: writing the cache
# costs a premium over base input price, reads are discounted to 10%.
CACHE_TTL_MIN = 5
WINDOWS_PER_DAY = int(24 * 60 / CACHE_TTL_MIN)   # 288 five-minute windows in a day

# EXTENSION 4 — the generator gives reasoning requests 6x the output tokens, and
# reasoning burns ~80x the energy per token (deck §11).
REASONING_OUTPUT_MULTIPLIER = 6


def cache_economics(rows) -> dict:
    """EXTENSION 3 — decide, per prefix group, whether prompt caching earns its keep.

    A "prefix group" is one (team, route_tier) pair: those requests share the same
    static system prompt, so they hit the same cache entry. The number that matters
    is re-reads *per write*, and that is bounded by the TTL: a 5-minute entry can
    only be re-read by requests that arrive inside the same 5-minute window. A
    low-traffic group re-writes the entry more often than it reads it back.
    """
    groups = defaultdict(lambda: {"n": 0, "cached_tok": 0, "input_tok": 0, "n_cached": 0})
    for r in rows:
        g = groups[(r["team"], r["route_tier"])]
        g["n"] += 1
        g["input_tok"] += int(num(r["input_tokens"]))
        c = int(num(r["cached_input_tokens"]))
        if c > 0:
            g["n_cached"] += 1
            g["cached_tok"] += c

    out = {}
    for key, g in groups.items():
        tier = key[1]
        base_price = MODEL_PRICES[tier][0]
        write_cost = base_price * pricing.CACHE_WRITE_PREMIUM
        be = pricing.break_even_cache_reads(write_cost, 0.10, base_price)
        # requests landing in one TTL window; the first writes, the rest re-read
        per_window = g["n"] / WINDOWS_PER_DAY
        reads_per_write = max(0.0, per_window - 1.0)
        worth = pricing.cache_is_worth_it(reads_per_write, write_cost, 0.10, base_price)
        out[key] = {
            "team": key[0], "tier": tier, "requests": g["n"],
            "cache_hit_frac": g["cached_tok"] / g["input_tok"] if g["input_tok"] else 0.0,
            "reqs_per_window": round(per_window, 2),
            "reads_per_write": round(reads_per_write, 2),
            "break_even_reads": round(be, 3),
            "naive_reads_per_day": max(0, g["n_cached"] - 1),   # ignoring TTL
            "worth_it": worth,
        }
    return out


def reasoning_budget(rows, cache_econ) -> dict:
    """EXTENSION 4 — split spend and energy by is_reasoning, then price a cap."""
    split = {0: {"n": 0, "cost": 0.0, "tokens": 0, "wh": 0.0},
             1: {"n": 0, "cost": 0.0, "tokens": 0, "wh": 0.0}}
    reasoning_rows = []
    for r in rows:
        k = int(num(r["is_reasoning"]))
        inp, out_t = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        gate = cache_econ[(r["team"], r["route_tier"])]["worth_it"]
        eff_cached = int(num(r["cached_input_tokens"])) if gate else 0
        cost = pricing.request_cost(inp, out_t, pin, pout,
                                    cached_in=eff_cached,
                                    batch=bool(int(num(r["is_batch"]))))
        wh = sustainability.wh_per_query(inp + out_t, is_reasoning=bool(k))
        s = split[k]
        s["n"] += 1
        s["cost"] += cost
        s["tokens"] += inp + out_t
        s["wh"] += wh
        if k:
            reasoning_rows.append({"row": r, "cost": cost, "wh": wh,
                                   "inp": inp, "out": out_t,
                                   "eff_cached": eff_cached,
                                   "latency": num(r["latency_ms"])})

    total_n = split[0]["n"] + split[1]["n"]
    total_cost = split[0]["cost"] + split[1]["cost"]
    total_wh = split[0]["wh"] + split[1]["wh"]

    # --- what a cap would save -------------------------------------------------
    # Downgrading a reasoning request means it answers without the long chain: the
    # output shrinks by REASONING_OUTPUT_MULTIPLIER and the 80x energy tax is gone.
    def cap_to(target_share: float) -> dict:
        keep = int(round(target_share * total_n))
        # keep reasoning where it plausibly earns its cost: the longest answers
        ranked = sorted(reasoning_rows, key=lambda x: -x["out"])
        downgraded = ranked[keep:]
        saved_cost = saved_wh = 0.0
        for d in downgraded:
            r = d["row"]
            short_out = max(1, int(d["out"] / REASONING_OUTPUT_MULTIPLIER))
            pin, pout = MODEL_PRICES[r["route_tier"]]
            new_cost = pricing.request_cost(d["inp"], short_out, pin, pout,
                                            cached_in=d["eff_cached"],
                                            batch=bool(int(num(r["is_batch"]))))
            new_wh = sustainability.wh_per_query(d["inp"] + short_out, is_reasoning=False)
            saved_cost += d["cost"] - new_cost
            saved_wh += d["wh"] - new_wh
        return {"target_share": target_share, "kept": min(keep, len(reasoning_rows)),
                "downgraded": len(downgraded),
                "saved_cost_daily": round(saved_cost, 2),
                "saved_wh_daily": round(saved_wh, 1),
                "saved_cost_pct": round(saved_cost / total_cost * 100, 1) if total_cost else 0.0,
                "saved_wh_pct": round(saved_wh / total_wh * 100, 1) if total_wh else 0.0}

    caps = [cap_to(s) for s in (0.05, 0.02, 0.01)]
    avg_lat_r = (sum(x["latency"] for x in reasoning_rows) / len(reasoning_rows)
                 if reasoning_rows else 0.0)

    return {
        "reasoning_requests": split[1]["n"], "normal_requests": split[0]["n"],
        "traffic_share": split[1]["n"] / total_n if total_n else 0.0,
        "cost_share": split[1]["cost"] / total_cost if total_cost else 0.0,
        "wh_share": split[1]["wh"] / total_wh if total_wh else 0.0,
        "token_share": split[1]["tokens"] / (split[0]["tokens"] + split[1]["tokens"]),
        "reasoning_cost_daily": round(split[1]["cost"], 2),
        "normal_cost_daily": round(split[0]["cost"], 2),
        "reasoning_wh_daily": round(split[1]["wh"], 1),
        "normal_wh_daily": round(split[0]["wh"], 1),
        "cost_per_req_reasoning": round(split[1]["cost"] / split[1]["n"], 5) if split[1]["n"] else 0.0,
        "cost_per_req_normal": round(split[0]["cost"] / split[0]["n"], 5) if split[0]["n"] else 0.0,
        "wh_per_req_reasoning": round(split[1]["wh"] / split[1]["n"], 2) if split[1]["n"] else 0.0,
        "wh_per_req_normal": round(split[0]["wh"] / split[0]["n"], 4) if split[0]["n"] else 0.0,
        "avg_latency_ms_reasoning": round(avg_lat_r),
        "caps": caps, "total_wh_daily": round(total_wh, 1),
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    cache_econ = cache_economics(rows)

    base_cost = opt_cost = 0.0
    # per-lever isolation: each lever applied alone, against the same baseline
    only_cascade = only_cache = only_batch = 0.0
    cache_denied_savings = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        lin, lout = MODEL_PRICES["large"]
        pin, pout = MODEL_PRICES[r["route_tier"]]

        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        base_cost += pricing.request_cost(inp, out, lin, lout)

        # EXTENSION 3: only bank the cache discount where re-reads recoup the write
        gate = cache_econ[(r["team"], r["route_tier"])]["worth_it"]
        eff_cached = cached if gate else 0

        # OPTIMIZED: cascade (route_tier) + gated prompt caching + batch API
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=eff_cached, batch=is_batch)

        # each lever alone
        only_cascade += pricing.request_cost(inp, out, pin, pout)
        only_cache += pricing.request_cost(inp, out, lin, lout, cached_in=eff_cached)
        only_batch += pricing.request_cost(inp, out, lin, lout, batch=is_batch)
        if not gate and cached:
            cache_denied_savings += (
                pricing.request_cost(inp, out, pin, pout, cached_in=0, batch=is_batch)
                - pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch))

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # incremental (waterfall) decomposition in the order a team would ship them
    step_cascade = base_cost - only_cascade
    after_cascade = only_cascade
    after_cascade_cache = 0.0
    after_all = opt_cost
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        gate = cache_econ[(r["team"], r["route_tier"])]["worth_it"]
        cached = int(num(r["cached_input_tokens"])) if gate else 0
        pin, pout = MODEL_PRICES[r["route_tier"]]
        after_cascade_cache += pricing.request_cost(inp, out, pin, pout, cached_in=cached)
    step_cache = after_cascade - after_cascade_cache
    step_batch = after_cascade_cache - after_all

    levers = {"cascade": round(step_cascade, 2), "caching": round(step_cache, 2),
              "batch": round(step_batch, 2)}
    isolated = {"cascade": round(base_cost - only_cascade, 2),
                "caching": round(base_cost - only_cache, 2),
                "batch": round(base_cost - only_batch, 2)}

    reasoning = reasoning_budget(rows, cache_econ)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n  lever contribution (incremental, in ship order):")
        for k, v in levers.items():
            print(f"    {k:10} ${v:>8,.2f}/day  ({v/base_cost*100:>4.1f} pts of the {savings_pct:.1f}% total)")
        print("  lever contribution (each applied ALONE vs baseline):")
        for k, v in isolated.items():
            print(f"    {k:10} ${v:>8,.2f}/day  ({v/base_cost*100:>4.1f}%)")
        print("  -> cascade dominates: routing 80% of traffic to a 15x cheaper model beats")
        print("     any discount you can negotiate on the expensive model.")

        print("\n-- EXTENSION 3: is prompt caching worth its write premium? --")
        be_small = pricing.break_even_cache_reads(
            MODEL_PRICES['small'][0] * pricing.CACHE_WRITE_PREMIUM, 0.10, MODEL_PRICES['small'][0])
        be_large = pricing.break_even_cache_reads(
            MODEL_PRICES['large'][0] * pricing.CACHE_WRITE_PREMIUM, 0.10, MODEL_PRICES['large'][0])
        print(f"  write premium {pricing.CACHE_WRITE_PREMIUM:.2f}x base, reads at 0.10x ->")
        print(f"  break-even re-reads: small tier {be_small:.3f}, large tier {be_large:.3f}")
        print("  (the ratio is price-independent: 0.25 / 0.90 -> one single reuse pays it back 3.6x)")
        print(f"\n  but a {CACHE_TTL_MIN}-minute TTL caps re-reads at whatever arrives in the same window:")
        print(f"    {'team':11}{'tier':7}{'reqs/day':>9}{'hit%':>7}{'req/window':>12}{'reads/write':>12}{'break-even':>11}{'verdict':>10}")
        for key in sorted(cache_econ, key=lambda k: -cache_econ[k]["reqs_per_window"]):
            e = cache_econ[key]
            print(f"    {e['team']:11}{e['tier']:7}{e['requests']:>9}{e['cache_hit_frac']*100:>6.0f}%"
                  f"{e['reqs_per_window']:>12.2f}{e['reads_per_write']:>12.2f}{e['break_even_reads']:>11.3f}"
                  f"{'CACHE' if e['worth_it'] else 'no cache':>10}")
        denied = [k for k, e in cache_econ.items() if not e["worth_it"]]
        print(f"  gate rejects {len(denied)}/{len(cache_econ)} groups: {sorted(denied)}")
        print(f"  savings correctly NOT claimed on those groups: ${cache_denied_savings:,.2f}/day")
        print("  INSIGHT: a group receiving <1 request per TTL window re-writes the entry more")
        print("  often than it reads it, so 'enable caching everywhere' quietly loses money.")

        print("\n-- EXTENSION 4: reasoning budget --")
        rb = reasoning
        print(f"  reasoning: {rb['reasoning_requests']} reqs = {rb['traffic_share']:.1%} of traffic")
        print(f"             but {rb['cost_share']:.1%} of spend and {rb['wh_share']:.1%} of energy")
        print(f"  {'':14}{'$/day':>10}{'Wh/day':>12}{'$/req':>10}{'Wh/req':>10}")
        print(f"  {'reasoning':14}{rb['reasoning_cost_daily']:>10,.2f}{rb['reasoning_wh_daily']:>12,.0f}"
              f"{rb['cost_per_req_reasoning']:>10.5f}{rb['wh_per_req_reasoning']:>10.2f}")
        print(f"  {'normal':14}{rb['normal_cost_daily']:>10,.2f}{rb['normal_wh_daily']:>12,.0f}"
              f"{rb['cost_per_req_normal']:>10.5f}{rb['wh_per_req_normal']:>10.4f}")
        ratio_c = (rb['cost_per_req_reasoning'] / rb['cost_per_req_normal']
                   if rb['cost_per_req_normal'] else 0)
        ratio_w = (rb['wh_per_req_reasoning'] / rb['wh_per_req_normal']
                   if rb['wh_per_req_normal'] else 0)
        print(f"  a reasoning request costs {ratio_c:.1f}x and burns {ratio_w:.0f}x the energy of a normal one")
        print(f"  (6x the output tokens x ~80x Wh/token = ~{6*80}x in the limit)")
        print("\n  if reasoning were capped to a share of traffic (longest answers kept):")
        print(f"    {'cap':>6}{'kept':>7}{'downgraded':>12}{'$ saved/day':>13}{'% spend':>9}{'Wh saved/day':>14}{'% energy':>10}")
        for c in rb["caps"]:
            print(f"    {c['target_share']:>5.0%}{c['kept']:>7}{c['downgraded']:>12}"
                  f"{c['saved_cost_daily']:>13,.2f}{c['saved_cost_pct']:>8.1f}%"
                  f"{c['saved_wh_daily']:>14,.0f}{c['saved_wh_pct']:>9.1f}%")
        print(f"  ROUTING RULE: reasoning is 100% eval-team traffic and averages "
              f"{rb['avg_latency_ms_reasoning']:,}ms.")
        print("  Gate it on task complexity, not on team default: run the cheap model first and")
        print("  escalate to reasoning only when its self-reported confidence is below threshold.")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "levers": levers, "levers_isolated": isolated,
        "cache_economics": {f"{k[0]}/{k[1]}": v for k, v in cache_econ.items()},
        "cache_denied_savings_daily": round(cache_denied_savings, 2),
        "reasoning": reasoning,
    }


if __name__ == "__main__":
    run()
