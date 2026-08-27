# NimbusAI - GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,422  
**Projected savings:** $12,711  (**47%**)

Unit economics - the number that survives a change in fleet size:

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Inference $/1M tokens | $6.488 | $1.251 | -81% |
| Inference $/day | $48.87 | $9.42 | -81% |
| Total fleet $/month | $27,133 | $14,422 | -47% |

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,183 |
| Purchasing (spot/reserved) | $9,788 |
| Right-size memory-bound GPUs | $1,140 |
| Kill idle GPUs | $600 |
| **Total** | **$12,711 (47%)** |

### How each lever works, and what it is worth

| Lever | $/mo | % of total | Mechanism | Evidence | Effort |
|---|---|---|---|---|---|
| Inference (cascade/cache/batch) | $1,183 | 9% | Route the 80% of traffic that does not need a frontier model to one 15x cheaper; batch the eval queue at -50%; cache shared prefixes at -90% on reads | $6.488 -> $1.251 per 1M tokens over 7,533,027 tokens/day | Low - routing rule + batch endpoint |
| Purchasing (spot/reserved) | $9,788 | 77% | Interruptible jobs to spot, priced with checkpoint overhead and expected rework; steady jobs to the reserved term their runtime can actually justify | $25,667 -> $15,879/mo across 8 jobs | Medium - checkpointing + a commitment decision |
| Right-size memory-bound GPUs | $1,140 | 9% | Move bandwidth-starved GPUs to the cheapest part that still clears their achieved bandwidth plus 25% headroom | 4 GPUs at MFU < 30%, billed on active hours only so the idle lever is not double-counted | High - migration + latency re-validation |
| Kill idle GPUs | $600 | 5% | Stop paying for GPUs that report under 10% utilization overnight | $20.00/day of measured idle time on 1 GPU(s) | Low - scheduler policy |
| Cache gate (a loss avoided, not a saving) | $0 | 0% | Refuse the cache lever where a 5-minute TTL means the prefix is rewritten more often than it is read back | 5/8 prefix groups rejected; $28/mo of apparent savings correctly not claimed | Low - one predicate |

## Root cause: how 98% GPU-Util and 19% MFU coexist

| GPU | Type | GPU-Util % | MFU | MBU | Verdict |
|---|---|---|---|---|---|
| gpu-h100-4 | H100 | 98.2% | 0.194 | 0.207 | **LIE** - busy, not productive |
| gpu-a10g-1 | A10G | 96.9% | 0.268 | 0.302 | **LIE** - busy, not productive |
| gpu-h100-1 | H100 | 95.2% | 0.408 | 0.44 | healthy |
| gpu-h100-0 | H100 | 94.4% | 0.417 | 0.446 | healthy |
| gpu-h100-2 | H100 | 94.3% | 0.401 | 0.423 | healthy |
| gpu-h100-3 | H100 | 93.1% | 0.427 | 0.444 | healthy |
| gpu-h100-5 | H100 | 61.1% | 0.261 | 0.271 | borderline |
| gpu-l4-0 | L4 | 40.0% | 0.302 | 0.328 | honestly under-used |
| gpu-a100-0 | A100 | 31.4% | 0.259 | 0.276 | honestly under-used |
| gpu-a100-1 | A100 | 28.0% | 0.236 | 0.247 | honestly under-used |
| gpu-a10g-0 | A10G | 25.0% | 0.218 | 0.235 | honestly under-used |

**What the two numbers actually measure.** `nvidia-smi utilization.gpu` is a *duty-cycle* counter: the fraction of sampling intervals in which at least one kernel was resident on the device. It says nothing about how much of the device that kernel used, so a single small kernel occupying one SM for the whole interval still reports 100%. MFU is a *throughput* ratio: achieved FLOP/s over the peak the silicon is sold on. One answers "is the GPU busy?", the other answers "is the GPU doing work I am paying for?" - and only the second question has a dollar sign in it.

**The mechanism on gpu-h100-4.** It reports 98.2% utilization while delivering 192 of 990 TFLOP/s (MFU 0.194). At $2.50/hr, 81% of that rate buys nothing: roughly **$1,451/month** on this single GPU pays for arithmetic capability that is never exercised. The device is occupied almost continuously and idle *inside* almost every interval - which is precisely the state a utilization dashboard cannot show you.

**Where the time actually goes.** Its MBU is 0.207, so it is not saturating HBM either. Achieved arithmetic intensity is 277 FLOP/byte against a H100 ridge point of 296 FLOP/byte (990 TFLOP/s over 3.35 TB/s). Sitting below *both* roofs at once is the signature of a latency- and occupancy-bound kernel rather than a bandwidth-bound one: small batches, short sequences, and per-kernel launch and synchronisation overhead leave the tensor cores waiting between bursts. That makes the first fix a serving-configuration fix - larger batches, continuous batching, sequence packing - and only then a hardware fix.

*A caveat the data forces, which the playbook does not mention:* across all 11 GPUs here MBU tracks MFU within 4-13%, so this telemetry cannot cleanly separate memory-bound from occupancy-bound. On real hardware you would confirm with Nsight (SM occupancy, DRAM throughput, achieved warps) before buying anything on the strength of it.

**The proof that utilization cannot price work.** gpu-h100-3 reports 93.1% utilization - *lower* than gpu-h100-4's 98.2% - at MFU 0.427, which is 2.2x the useful work for the same $2.50/hr. Sharper still, gpu-a10g-1 and gpu-a10g-0 are the same part at the same price: 96.9% against 25.0% reported utilization (3.9x) for MFU 0.268 against 0.218 (only 1.23x). A dashboard ranking those two by utilization would rank them almost backwards.

**Consequence for governance.** GPU-Util is a liveness signal - useful for spotting a crashed trainer, worthless for capacity planning. Efficiency reviews and chargeback should run on MFU/MBU and on $/1M-token, neither of which can be gamed by keeping a device nominally busy.


## Recommendations, ordered by return on effort

| # | Action | Savings/mo | Effort | Payback | Risk |
|---|---|---|---|---|---|
| 1 | Move interruptible jobs to spot; commit the steady fleet to a reserved term | $9,788 | 15d ($653/eng-day) | 1.8 mo | A 3yr commitment is a liability if the roadmap moves; spot adds ~1.05x billed hours per useful hour |
| 2 | Shut GPUs down outside working hours | $600 | 2d ($300/eng-day) | 4.0 mo | An overnight submission waits for the morning window unless there is a wake-on-demand path |
| 3 | Ship the cascade router, the batch queue and the gated prompt cache | $1,183 | 10d ($118/eng-day) | 10.1 mo | Cascading can degrade answers; needs an eval gate on the escalation rate |
| 4 | Re-provision the bandwidth-starved GPUs onto cheaper parts | $1,140 | 30d ($38/eng-day) | 31.6 mo | Halving bandwidth headroom can breach the p95 latency SLO; must be benchmarked before cutover |

Ordering is savings per unit of engineering effort, not savings alone: items 1-2 are configuration changes that ship in a sprint, the tail requires migration work and should not block them.

- **Ship the cascade router, the batch queue and the gated prompt cache** - the cascade alone is 95% of this lever - ship the router first and negotiate discounts second
- **Re-provision the bandwidth-starved GPUs onto cheaper parts** - the payback here is longer than a year of the engineering time it costs, so it belongs at the next hardware refresh - fix the batch size first, which is free and may remove the need entirely

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1
  - cleanest grid (lowest gCO2/kWh): **europe-north1**
  - cheapest electricity ($/kWh): **us-east-wa**
  - best joint pick (carbon + price + latency): **us-east-wa**
  - on this catalogue the cheapest grid and the cleanest grid are *not* the same place, so "cheapest+cleanest" has to be split into the three picks above; see the carbon section for the scoring.
- Energy per *reasoning* query: 19.2 Wh (80x a normal query)

### Carbon, converted into a line on the electricity bill

The fleet draws **6,602 kWh/month** (PUE 1.12), of which **4,734 kWh** belongs to interruptible jobs that are free to move.

| Region | $/kWh | gCO2/kWh | Electricity $/mo | tCO2e/mo | RTT | Balance score |
|---|---|---|---|---|---|---|
| us-east-wa | 0.055 | 90 | $260 | 0.43 | 60ms | 0.10 |
| us-west-2 | 0.070 | 120 | $331 | 0.57 | 70ms | 0.26 |
| europe-north1 | 0.090 | 30 | $426 | 0.14 | 110ms | 0.28 |
| us-east-1 | 0.120 | 380 | $568 | 1.80 | 15ms | 1.08 |
| europe-central2 | 0.180 | 660 | $852 | 3.12 | 120ms | 2.00 |

Moving the 5 interruptible jobs from us-east-1 to europe-north1 cuts **1,657 kgCO2e/month** (92%, about 19.9 tonnes/year) and lowers the electricity line by $142/mo. The three "best regions" are not the same place: cleanest is **europe-north1** (30 gCO2/kWh), cheapest electricity is **us-east-wa** ($0.055/kWh), and the best joint pick is **us-east-wa** - 3.0x the cleanest grid's carbon but still 76% below us-east-1, at 39% lower electricity cost and roughly half the added latency.

**Keeping this honest about money.** At 0.120 $/kWh the movable load's electricity is about $568/mo - and a neocloud has already priced that into the GPU-hour rate. So region choice is a large *carbon* lever and a small *cost* lever: the biggest available electricity saving here is $308/mo, which only reaches the income statement if you self-host or negotiate a region-differentiated rate. Reported as a carbon-intensity target, it is worth 92% on the movable half of the fleet.

**The other half of the energy story is inference, not hardware.** Serving draws 31,675 Wh/day (950 kWh/mo). Reasoning requests are 8.4% of traffic but 94.0% of that energy, at 148.2 Wh/request against 0.8584 Wh for a normal one (173x). Gating reasoning on task complexity is therefore a sustainability lever as much as a cost one - capping it at 5% of traffic saves 7,880 Wh/day (25% of serving energy).

**Latency is the real constraint.** The clean grids are the far ones: europe-north1 is ~110ms from a US-East user against 15ms at home. That is acceptable for the interruptible training and eval jobs above, and unacceptable for the 24/7 inference jobs - 28% of fleet energy therefore stays where it is. Carbon-aware scheduling is a lever on the movable half only, and any report claiming otherwise is quietly proposing an SLO breach.

## Extensions built, with measured results

**1. Purchasing policy that prices risk and term** - `recommend_tier` now scores every feasible tier in $/*useful*-hour: spot carries a per-GPU-type reclaim rate plus checkpoint overhead and expected rework, a reservation is charged for all 24h/day because that is what a commitment bills, and 1yr vs 3yr is gated on how long the job actually runs.  
Result: The original duty-cycle policy claimed 39.1% savings; pricing reservations round the clock revises that **down** to 38.1% ($+252/mo). The break-even reclaim rate for every GPU in the catalogue is above 100%/h, so spot is not the risky choice for interruptible work - the under-utilised commitment is.

**2. Right-sizing justified by bandwidth, not by sticker price** - For each GPU under 30% MFU, find the cheapest catalogue part that still delivers the bandwidth it actually uses plus 25% headroom, and price the swap in $/TB/s-hr and $/GB-VRAM-hr rather than $/GPU-hr.  
Result: 4 of 11 GPUs qualify, worth **$1,140/mo** billed on active hours only. Picking on $/GPU-hr instead would have proposed an L4 for the H100 workloads, which cannot sustain their memory traffic.

**3. Prompt caching gated on its own break-even** - A cache write costs a 25% premium and a read saves 90%, so break-even is 0.25/0.90 = 0.278 re-reads. But a 5-minute TTL caps re-reads at whatever arrives inside the same window, so the gate is evaluated per (team, route_tier) prefix group.  
Result: 5/8 groups receive under one request per TTL window and therefore *lose* money on caching: assistant/large, eval/large, eval/small, rag/large, search/large. Declining them forgoes $28/mo of apparent savings that would not have been real. "Enable caching everywhere" is the wrong default.

**4. Reasoning-token budget** - Split spend and energy by `is_reasoning`, then simulate capping reasoning to a share of traffic by keeping the longest answers and downgrading the rest.  
Result: Reasoning is 8.4% of requests but 14.9% of spend and 94.0% of energy. Capping to 5% saves $0.23/day (2.4% of the bill) and 7,880 Wh/day (25%).

**5. Carbon-aware scheduling** - Place only the interruptible jobs by grid carbon intensity, and score all five regions on price, carbon and latency together.  
Result: **1,657 kgCO2e/mo** (92%) off the movable load by moving it to europe-north1; us-east-wa is the better joint pick. 28% of fleet energy is latency-bound and cannot move.


## Assumptions and limits

- **The two datasets are separate views.** `gpu_telemetry.csv` (11 GPUs) and `workloads.csv` (8 jobs) are not reconciled to the same physical fleet, so the four levers are summed as if independent. In production they interact: the purchasing lever would apply to *right-sized* rates, which would make the combined figure somewhat lower than the sum shown here. The per-lever numbers are the defensible ones; the total is an upper bound.
- **Idle and right-size do not double-count.** One GPU (gpu-h100-5) is both idle overnight and a right-size candidate, so its right-size saving is billed on active hours (16/day) rather than 24.
- **Baseline is a naive deployment, not today's bill.** Inference baseline prices every request on the large model with no cache and no batching ($48.87/day); purchasing baseline is 100% on-demand ($25,667/mo). Savings against a partly-optimised estate would be smaller.
- **Payback uses $1,200/engineer-day loaded cost.** Change that number and the ranking of the last two actions changes with it; the ranking of the first two does not.
- **Prices, carbon intensities and the 80x reasoning-energy multiplier are illustrative June-2026 snapshots.** GPU pricing moves fast enough that a purchasing commitment should be re-baselined before it is signed.

_Figures are June-2026 as-of snapshots; re-baseline before acting._