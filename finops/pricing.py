"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# ---------------------------------------------------------------------------
# EXTENSION 1 — purchasing policy that prices risk and commitment term
# ---------------------------------------------------------------------------
# Reclaim risk is NOT uniform across accelerators. Scarce high-demand parts get
# reclaimed less often (the spot pool is thin but stable, because buyers who need
# them pay full rate); cheap commodity inference parts churn far more.
SPOT_INTERRUPT_RATE = {
    "H100": 0.03,
    "H200": 0.04,
    "B200": 0.06,
    "A100": 0.05,
    "MI300X": 0.05,
    "L4": 0.10,
    "A10G": 0.12,   # commodity inference GPU: cheapest spot, churniest pool
}
DEFAULT_INTERRUPT_RATE = 0.05

# A reservation bills 24/7 for the whole term whether or not you run anything on
# it. Commit only when the workload will still be there.
COMMIT_MIN_DAYS_3YR = 21
COMMIT_MIN_DAYS_1YR = 7


def spot_interrupt_rate(gpu_type: str | None = None) -> float:
    """Per-hour reclaim probability for a GPU type (the H100 pool is calmer than A10G)."""
    return SPOT_INTERRUPT_RATE.get(gpu_type or "", DEFAULT_INTERRUPT_RATE)


def spot_effective_multiplier(
    interrupt_rate: float,
    ckpt_overhead_frac: float = 0.03,
    rework_hours_per_interrupt: float = 0.5,
) -> float:
    """Billed spot hours per hour of *useful* work.

    Checkpointing costs a steady overhead; each reclaim throws away the work done
    since the last checkpoint. Always > 1.0 — this is the true price of a spot hour.
    """
    return (1.0 + ckpt_overhead_frac) + max(0.0, interrupt_rate) * rework_hours_per_interrupt


def break_even_interrupt_rate(
    spot_hr: float,
    on_demand_hr: float,
    ckpt_overhead_frac: float = 0.03,
    rework_hours_per_interrupt: float = 0.5,
) -> float:
    """Reclaim rate at which spot stops beating on-demand.

    Solves spot_hr * ((1+ckpt) + r*rework) = on_demand_hr for r. A large answer
    means "spot wins even in a hostile pool" — the usual verdict for LLM GPUs.
    """
    if spot_hr <= 0 or rework_hours_per_interrupt <= 0:
        return 0.0
    return max(0.0, (on_demand_hr / spot_hr - (1.0 + ckpt_overhead_frac)) / rework_hours_per_interrupt)


def commit_term(job_days: float | None) -> str | None:
    """Which reserved term the workload's real duration can justify."""
    if job_days is None:
        return "3yr"                      # legacy default: assume steady state
    if job_days >= COMMIT_MIN_DAYS_3YR:
        return "3yr"
    if job_days >= COMMIT_MIN_DAYS_1YR:
        return "1yr"
    return None                           # too short to commit to anything


def effective_hourly_cost(
    tier: str,
    hours_per_day: float,
    prices: dict,
    gpu_type: str | None = None,
    term: str = "3yr",
) -> float:
    """USD per hour of *useful* work for a tier — the only comparable unit.

    The subtlety a naive model misses: on-demand and spot bill per hour used, but a
    reservation bills all 24 hours of every day. A reserved GPU running 3h/day
    therefore costs 8x its sticker rate per useful hour. This is *why* the
    break-even utilization is 1 - discount rather than a rule of thumb.
    """
    hpd = max(0.0, min(24.0, hours_per_day))
    if hpd <= 0:
        return float("inf")
    if tier == "on_demand":
        return float(prices.get("on_demand_hr", 0.0))
    if tier == "spot":
        rate = float(prices.get("spot_hr", 0.0))
        return rate * spot_effective_multiplier(spot_interrupt_rate(gpu_type))
    if tier == "reserved":
        key = "reserved_1yr_hr" if term == "1yr" else "reserved_3yr_hr"
        rate = float(prices.get(key, prices.get("reserved_3yr_hr", 0.0)))
        return rate * (24.0 / hpd)        # a commitment bills round the clock
    return float("inf")


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
    prices: dict | None = None,
) -> str:
    """Pick a purchasing tier for a workload.

    Called with only (hours_per_day, interruptible) this keeps the DOCUMENTED simple
    policy — duty cycle vs break-even, spot for anything interruptible:
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    EXTENSION 1: pass `prices` (a price_catalog row) to get the economic policy,
    which prices reclaim risk per GPU type and the 24/7 nature of a commitment
    instead of trusting the duty-cycle heuristic. See recommend_tier_detailed().
    """
    if prices is None:
        duty = max(0.0, hours_per_day) / 24.0
        be = break_even_utilization(reserved_discount)
        if interruptible and hours_per_day < 24:
            return "spot"
        if duty >= be:
            return "reserved"
        return "on_demand"
    return recommend_tier_detailed(
        hours_per_day, interruptible, reserved_discount, gpu_type, job_days, prices
    )["tier"]


def recommend_tier_detailed(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
    prices: dict | None = None,
) -> dict:
    """EXTENSION 1 — cost-minimising tier choice with a written rationale.

    Scores every *feasible* tier in $/useful-hour and takes the cheapest:
      - spot is feasible only for interruptible work, and is charged its true
        multiplier (checkpoint overhead + expected rework at this GPU's reclaim rate)
      - reserved is feasible only if the job runs long enough to justify a term,
        and is charged for 24h/day because that is what a commitment bills
      - on-demand is always feasible

    Returns the tier ('spot' | 'reserved' | 'on_demand'), the commitment term, the
    winning $/useful-hour, every candidate score, and the reason.
    """
    if prices is None:
        tier = recommend_tier(hours_per_day, interruptible, reserved_discount)
        return {"tier": tier, "term": "3yr" if tier == "reserved" else None,
                "effective_hr": None, "candidates": {},
                "interrupt_rate": spot_interrupt_rate(gpu_type),
                "reason": "documented simple policy"}

    term = commit_term(job_days)
    rate = spot_interrupt_rate(gpu_type)
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)

    candidates = {"on_demand": effective_hourly_cost("on_demand", hours_per_day, prices, gpu_type)}
    if interruptible:
        candidates["spot"] = effective_hourly_cost("spot", hours_per_day, prices, gpu_type)
    if term is not None:
        candidates["reserved"] = effective_hourly_cost(
            "reserved", hours_per_day, prices, gpu_type, term=term)

    tier = min(candidates, key=candidates.get)
    if tier == "spot":
        reason = (f"interruptible; {gpu_type or 'GPU'} reclaim ~{rate:.0%}/h means only "
                  f"{spot_effective_multiplier(rate):.2f}x billed hours per useful hour")
    elif tier == "reserved":
        days = f"{job_days:.0f}d" if job_days is not None else "steady-state"
        reason = f"{duty:.0%} duty over {days} clears the {be:.0%} break-even -> {term} commit"
    elif term is None and job_days is not None:
        reason = f"only {job_days:.0f}d of runtime - too short to commit, and not interruptible"
    else:
        reason = f"{duty:.0%} duty sits below the {be:.0%} break-even"

    return {"tier": tier, "term": term if tier == "reserved" else None,
            "effective_hr": round(candidates[tier], 4),
            "candidates": {k: round(v, 4) for k, v in candidates.items()},
            "interrupt_rate": rate, "reason": reason}


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


# ---------------------------------------------------------------------------
# EXTENSION 3 — is prompt caching actually worth it?
# ---------------------------------------------------------------------------
# Caching is not free: providers either charge a premium to WRITE the cache
# (Anthropic ~1.25x base input for the 5-min TTL) or bill storage per hour
# (Gemini). A cached prefix only pays for itself once it is re-read enough times.
CACHE_WRITE_PREMIUM = 0.25   # extra cost to write, as a fraction of base input price


def break_even_cache_reads(
    write_cost_per_m: float,
    read_discount: float = 0.10,
    base_price_per_m: float = 1.0,
) -> float:
    """How many cache re-reads are needed before caching pays for its write.

    Each read saves (1 - read_discount) x base price; the write costs
    write_cost_per_m once, so break-even reads = write / saving-per-read.
    Units follow base_price_per_m (default 1.0 = "multiples of the base price").
    """
    saving_per_read = (1.0 - read_discount) * base_price_per_m
    if saving_per_read <= 0:
        return float("inf")
    return max(0.0, write_cost_per_m) / saving_per_read


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
    base_price_per_m: float = 1.0,
) -> bool:
    """True when re-reads recoup the cost of writing the cache.

    avg_cache_reads is the mean number of times a cached prefix is read back.
    Anthropic-style economics (write premium 0.25x, reads at 0.10x) break even
    below a single re-read, so caching a shared system prompt is nearly free money
    — but a prefix written and never re-read is a pure loss.
    """
    return avg_cache_reads > break_even_cache_reads(
        write_cost_per_m, read_discount, base_price_per_m)
