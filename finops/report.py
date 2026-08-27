"""Report assembly - the lab's deliverable: baseline vs optimized + savings chart.

build_report() keeps its original 5-argument contract; every analysis section is an
optional keyword, so the report degrades gracefully to the minimal version.
"""
from __future__ import annotations


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 unit_economics: dict | None = None,
                 lever_detail: list | None = None,
                 root_cause: dict | None = None,
                 actions: list | None = None,
                 carbon: dict | None = None,
                 extensions: list | None = None,
                 assumptions: list | None = None) -> str:
    """Return a markdown cost-optimization report.

    Required: baseline, optimized, per-lever savings. Everything else is optional
    analysis: unit economics, lever mechanisms, the root cause of the GPU-Util lie,
    an ROI-ordered action list, extended carbon economics, and extension results.
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI - GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
    ]
    if unit_economics:
        lines += [
            "",
            "Unit economics - the number that survives a change in fleet size:",
            "",
            "| Metric | Baseline | Optimized | Change |",
            "|---|---|---|---|",
        ]
        for row in unit_economics.get("rows", []):
            lines.append(f"| {row['metric']} | {row['before']} | {row['after']} | {row['change']} |")

    lines += [
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    if baseline_usd > 0:
        lines.append(f"| **Total** | **${savings:,.0f} ({pct:.0f}%)** |")

    # ---- per-lever mechanism / evidence / effort -----------------------------
    if lever_detail:
        lines += [
            "",
            "### How each lever works, and what it is worth",
            "",
            "| Lever | $/mo | % of total | Mechanism | Evidence | Effort |",
            "|---|---|---|---|---|---|",
        ]
        for d in lever_detail:
            share = d["usd"] / savings * 100 if savings else 0.0
            lines.append(
                f"| {d['name']} | ${d['usd']:,.0f} | {share:.0f}% | {d['mechanism']} "
                f"| {d['evidence']} | {d['effort']} |")

    # ---- C.2: root cause of the GPU-Util lie --------------------------------
    if root_cause:
        lines += ["", f"## {root_cause.get('title', 'Root cause')}", ""]
        if root_cause.get("evidence_table"):
            lines += ["| GPU | Type | GPU-Util % | MFU | MBU | Verdict |", "|---|---|---|---|---|---|"]
            for e in root_cause["evidence_table"]:
                lines.append(f"| {e['gpu_id']} | {e['gpu_type']} | {e['util']}% | {e['mfu']} "
                             f"| {e['mbu']} | {e['verdict']} |")
            lines.append("")
        for para in root_cause.get("body", []):
            lines += [para, ""]

    # ---- C.2: ROI-ordered recommendations ----------------------------------
    if actions:
        lines += [
            "",
            "## Recommendations, ordered by return on effort",
            "",
            "| # | Action | Savings/mo | Effort | Payback | Risk |",
            "|---|---|---|---|---|---|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(f"| {i} | {a['action']} | ${a['usd']:,.0f} | {a['effort']} "
                         f"| {a['payback']} | {a['risk']} |")
        lines += ["", "Ordering is savings per unit of engineering effort, not savings alone: "
                      "items 1-2 are configuration changes that ship in a sprint, the tail "
                      "requires migration work and should not block them.", ""]
        for note in [a for a in actions if a.get("note")]:
            lines.append(f"- **{note['action']}** - {note['note']}")

    # ---- C.1 + C.2: sustainability, priced ---------------------------------
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}",
        ]
        if sustainability.get("cheapest_region"):
            lines += [
                f"  - cleanest grid (lowest gCO2/kWh): "
                f"**{sustainability.get('best_region')}**",
                f"  - cheapest electricity ($/kWh): "
                f"**{sustainability['cheapest_region']}**",
                f"  - best joint pick (carbon + price + latency): "
                f"**{sustainability.get('balanced_region')}**",
                "  - on this catalogue the cheapest grid and the cleanest grid are *not* the "
                "same place, so \"cheapest+cleanest\" has to be split into the three picks "
                "above; see the carbon section for the scoring.",
            ]
        if sustainability.get("wh_per_query_reasoning") is not None:
            lines.append(f"- Energy per *reasoning* query: "
                         f"{sustainability['wh_per_query_reasoning']:.1f} Wh "
                         f"({sustainability.get('reasoning_multiplier', 80):.0f}x a normal query)")
    if carbon:
        lines += [
            "",
            "### Carbon, converted into a line on the electricity bill",
            "",
            f"The fleet draws **{carbon['fleet_kwh']:,.0f} kWh/month** (PUE {carbon['pue']}), of which "
            f"**{carbon['movable_kwh']:,.0f} kWh** belongs to interruptible jobs that are free to move.",
            "",
            "| Region | $/kWh | gCO2/kWh | Electricity $/mo | tCO2e/mo | RTT | Balance score |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in carbon["region_table"]:
            lines.append(f"| {r['region']} | {r['usd_per_kwh']:.3f} | {r['gco2_per_kwh']} "
                         f"| ${r['energy_cost']:,.0f} | {r['carbon_kg']/1000:.2f} "
                         f"| {r['rtt_ms']}ms | {r['balance_score']:.2f} |")
        for para in carbon.get("body", []):
            lines += ["", para]

    # ---- extensions ---------------------------------------------------------
    if extensions:
        lines += ["", "## Extensions built, with measured results", ""]
        for e in extensions:
            lines.append(f"**{e['name']}** - {e['what']}  ")
            lines.append(f"Result: {e['result']}")
            lines.append("")

    if assumptions:
        lines += ["", "## Assumptions and limits", ""]
        for a in assumptions:
            lines.append(f"- {a}")

    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, baseline_usd: float | None = None,
                      title: str = "GPU cost savings by FinOps lever") -> str:
    """Write a savings waterfall PNG. Returns the path. No-op if matplotlib absent.

    With `baseline_usd` this is a true waterfall: it opens at the baseline bill, steps
    down once per lever (each bar floating between the running total before and after
    that lever), and closes on the optimized bill. Without it, it degrades to a plain
    bar chart of the same numbers.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    names = list(levers.keys())
    vals = [float(levers[n]) for n in names]

    if baseline_usd is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(names, vals, color="#2e548a")
        ax.set_ylabel("Savings (USD / month)")
        ax.set_title(title)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    total = sum(vals)
    optimized = baseline_usd - total
    labels = ["Baseline"] + [n.split(" (")[0] for n in names] + ["Optimized"]

    # bottoms/heights: full bars for the two endpoints, floating bars for each step
    bottoms = [0.0]
    heights = [baseline_usd]
    running = baseline_usd
    for v in vals:
        bottoms.append(running - v)
        heights.append(v)
        running -= v
    bottoms.append(0.0)
    heights.append(optimized)

    colors = ["#37474f"] + ["#c0392b"] * len(vals) + ["#1e8449"]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(x, heights, bottom=bottoms, color=colors, width=0.62, zorder=3)

    # connectors: the top of each bar is the floor the next one starts from
    tops = [bottoms[i] + heights[i] for i in range(len(labels))]
    for i in range(len(labels) - 1):
        y = tops[i] if i == 0 else bottoms[i]
        ax.plot([i + 0.31, i + 1 - 0.31], [y, y], color="#90a4ae",
                lw=1.1, ls="--", zorder=2)

    for i, (b, h) in enumerate(zip(bottoms, heights)):
        if i == 0 or i == len(labels) - 1:
            ax.text(i, b + h + baseline_usd * 0.015, f"${h:,.0f}",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        else:
            share = h / total * 100 if total else 0.0
            ax.text(i, b + h + baseline_usd * 0.015,
                    f"-${h:,.0f}\n({share:.0f}% of savings)",
                    ha="center", va="bottom", fontsize=8.5, color="#c0392b")

    pct = total / baseline_usd * 100 if baseline_usd else 0.0
    ax.set_ylabel("Monthly spend (USD)")
    ax.set_title(f"{title}\nbaseline ${baseline_usd:,.0f} -> optimized ${optimized:,.0f} "
                 f"({pct:.0f}% saved, ${total:,.0f}/mo)", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0, baseline_usd * 1.18)
    ax.grid(axis="y", color="#eceff1", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
