"""Pure-stdlib Prometheus exporter: turns synthetic GPU telemetry into real-time FinOps cost metrics.

Metrics exposed:
- `gpu_util_pct`: nvidia-smi time-active clock (%)
- `gpu_mfu`: Model FLOPs Utilization (real compute efficiency 0..1)
- `gpu_mbu`: Model Bandwidth Utilization (real memory bandwidth efficiency 0..1)
- `gpu_hourly_cost_usd`: On-demand $/GPU-hr rate
- `gpu_wasted_cost_usd_per_hr`: Money paid for unachieved FLOPs: `(1 - MFU) * $/hr`
- `gpu_idle_waste_usd_per_hr`: Money paid when GPU is completely idle (<10% util)
- `gpu_power_watts`: GPU power draw in Watts
- `gpu_carbon_rate_gco2_per_hr`: Grid carbon footprint rate in gCO2e/hour (us-east-1 grid snapshot)

Runnable with or without Docker:
    python bonus/docker/exporter.py
Then scrape: http://localhost:9101/metrics
"""
from __future__ import annotations
import csv
import os
import sys
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from finops import metrics, sustainability

DATA_DIR = os.environ.get(
    "LAB_DATA_DIR",
    os.path.join(ROOT, "data"),
)


def _load():
    cat = {}
    with open(os.path.join(DATA_DIR, "price_catalog.csv")) as f:
        for r in csv.DictReader(f):
            cat[r["gpu_type"]] = r

    agg = defaultdict(lambda: {
        "util": [],
        "mfu": [],
        "mbu": [],
        "power": [],
        "idle_hours": 0,
        "type": None,
    })

    with open(os.path.join(DATA_DIR, "gpu_telemetry.csv")) as f:
        for r in csv.DictReader(f):
            gid = r["gpu_id"]
            gtype = r["gpu_type"]
            a = agg[gid]
            a["type"] = gtype
            peak_fp16 = float(cat[gtype]["peak_tflops_fp16"]) or 1.0
            peak_bw = float(cat[gtype]["peak_bw_tbs"]) or 1.0

            achieved_flops = float(r["achieved_tflops"])
            achieved_bw = float(r["achieved_bw_tbs"])
            util = float(r["gpu_util_pct"])
            power = float(r["power_w"])

            a["util"].append(util)
            a["mfu"].append(metrics.compute_mfu(achieved_flops, peak_fp16))
            a["mbu"].append(metrics.compute_mbu(achieved_bw, peak_bw))
            a["power"].append(power)
            if util < 10.0:
                a["idle_hours"] += 1

    return cat, agg


def render() -> str:
    cat, agg = _load()
    out = [
        "# HELP gpu_util_pct nvidia-smi time-active clock utilization (0-100%)",
        "# TYPE gpu_util_pct gauge",
        "# HELP gpu_mfu Model FLOPs Utilization - real compute efficiency (0..1)",
        "# TYPE gpu_mfu gauge",
        "# HELP gpu_mbu Model Bandwidth Utilization - real memory bandwidth efficiency (0..1)",
        "# TYPE gpu_mbu gauge",
        "# HELP gpu_hourly_cost_usd on-demand rate in USD per GPU-hour",
        "# TYPE gpu_hourly_cost_usd gauge",
        "# HELP gpu_wasted_cost_usd_per_hr money paid for unachieved FLOPs (1-MFU)*cost",
        "# TYPE gpu_wasted_cost_usd_per_hr gauge",
        "# HELP gpu_idle_waste_usd_per_hr money wasted while GPU sits idle (<10% util)",
        "# TYPE gpu_idle_waste_usd_per_hr gauge",
        "# HELP gpu_power_watts average GPU power consumption in Watts",
        "# TYPE gpu_power_watts gauge",
        "# HELP gpu_carbon_rate_gco2_per_hr estimated carbon emissions rate (gCO2/hr in us-east-1)",
        "# TYPE gpu_carbon_rate_gco2_per_hr gauge",
    ]

    for gid, a in sorted(agg.items()):
        gtype = a["type"]
        util = sum(a["util"]) / len(a["util"])
        mfu = sum(a["mfu"]) / len(a["mfu"])
        mbu = sum(a["mbu"]) / len(a["mbu"])
        power = sum(a["power"]) / len(a["power"])
        cost = float(cat[gtype]["on_demand_hr"])
        wasted_flops_cost = (1.0 - mfu) * cost

        # Idle fraction over the 24-hr sampling window
        idle_frac = a["idle_hours"] / len(a["util"]) if a["util"] else 0.0
        idle_cost_hr = idle_frac * cost

        # Carbon calculation (PUE 1.12 standard data center)
        kwh_per_hr = (power * 1.12) / 1000.0
        carbon_rate = sustainability.carbon_g(kwh_per_hr * 1000.0, region="us-east-1")

        lbl = f'{{gpu_id="{gid}",gpu_type="{gtype}"}}'
        out.append(f"gpu_util_pct{lbl} {util:.2f}")
        out.append(f"gpu_mfu{lbl} {mfu:.4f}")
        out.append(f"gpu_mbu{lbl} {mbu:.4f}")
        out.append(f"gpu_hourly_cost_usd{lbl} {cost:.2f}")
        out.append(f"gpu_wasted_cost_usd_per_hr{lbl} {wasted_flops_cost:.4f}")
        out.append(f"gpu_idle_waste_usd_per_hr{lbl} {idle_cost_hr:.4f}")
        out.append(f"gpu_power_watts{lbl} {power:.1f}")
        out.append(f"gpu_carbon_rate_gco2_per_hr{lbl} {carbon_rate:.2f}")

    return "\n".join(out) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = render().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_render():
    """Quick sanity test rendering output."""
    output = render()
    assert "gpu_util_pct" in output
    assert "gpu_mfu" in output
    assert "gpu_mbu" in output
    assert "gpu_wasted_cost_usd_per_hr" in output
    print(f"Metrics rendered successfully ({len(output.splitlines())} lines).")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_render()
    else:
        port = int(os.environ.get("PORT", "9101"))
        print(f"GPU FinOps Prometheus exporter listening on http://localhost:{port}/metrics (data: {DATA_DIR})")
        print("Press Ctrl+C to stop.")
        try:
            HTTPServer(("0.0.0.0", port), Handler).serve_forever()
        except KeyboardInterrupt:
            print("\nExporter stopped.")
