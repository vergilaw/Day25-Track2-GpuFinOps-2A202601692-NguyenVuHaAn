"""Run all five missions in order (M1 -> M5)."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from missions import (m1_efficiency_audit, m2_inference_levers, m3_purchasing,
                      m4_allocation, ext5_carbon_scheduling, m5_report)


def main():
    # ext5 is the carbon-aware scheduler (Extension 5); it runs before M5 because M5
    # pulls its region economics into the report's sustainability section.
    for m in (m1_efficiency_audit, m2_inference_levers, m3_purchasing, m4_allocation,
              ext5_carbon_scheduling):
        m.run(verbose=True)
        print()
    m5_report.run(verbose=True)


if __name__ == "__main__":
    main()
