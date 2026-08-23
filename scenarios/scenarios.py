"""
Drishti - Scenario Definitions & Execution Runner
===================================================
Defines canonical geopolitical shock scenarios for Phase 1 new-event inference.
Users can select a scenario via CLI argument (--scenario <name>) or modify the ACTIVE_SCENARIO.

Usage:
    python scenarios/scenarios.py --scenario wheat_russia_conflict
    python scenarios/scenarios.py --scenario indonesia_palm_oil
    python scenarios/scenarios.py --list
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# =========================================================================
# PREDEFINED SCENARIOS
# =========================================================================
# To add, enable, or modify scenarios, update the dictionary below.
SCENARIOS = {
    # ---------------------------------------------------------------------
    # Scenario 1: Russia-Ukraine Conflict Escalation (Wheat Import to India)
    # ---------------------------------------------------------------------
    "wheat_russia_conflict": {
        "description": "Geopolitical conflict escalation disrupting Russian wheat trade flows to India",
        "event_date": "2024-06-15",
        "event_country": "RUSSIA",
        "event_root": "18",               # 18 = Assault / Conflict
        "event_code": "180",              # 180 = Use unconventional mass violence / military assault
        "goldstein_score": -9.0,
        "avg_tone": -6.5,
        "num_mentions": 150,
        "commodity": "Wheat",
        "hs4": 1001,
        "trade_type": "Import",
        "shock_intensity": 1.5,
        "trade_share": 5.0,
    },

    # ---------------------------------------------------------------------
    # Scenario 2: Indonesia Palm Oil Export Restrictions & Levy (Import to India)
    # ---------------------------------------------------------------------
    "indonesia_palm_oil": {
        "description": "Indonesian export levy adjustments and domestic market obligation on palm oil",
        "event_date": "2024-06-15",
        "event_country": "INDONESIA",
        "event_root": "16",               # 16 = Reduce Relations / Trade Barrier
        "event_code": "163",              # 163 = Embargo, boycott, or export restriction
        "goldstein_score": -5.0,
        "avg_tone": -4.2,
        "num_mentions": 85,
        "commodity": "Palm Oil",
        "hs4": 1511,
        "trade_type": "Import",
        "shock_intensity": 1.5,
        "trade_share": 12.0,
    },

    # ---------------------------------------------------------------------
    # Scenario 3: India Wheat Export Restrictions to Bangladesh
    # ---------------------------------------------------------------------
    "india_wheat_export": {
        "description": "Indian export quota restrictions and regional food security calibration to Bangladesh",
        "event_date": "2024-06-15",
        "event_country": "BANGLADESH",
        "event_root": "16",               # 16 = Reduce Relations
        "event_code": "163",              # 163 = Export restriction
        "goldstein_score": -4.5,
        "avg_tone": -3.8,
        "num_mentions": 60,
        "commodity": "Wheat",
        "hs4": 1001,
        "trade_type": "Export",
        "shock_intensity": 1.2,
        "trade_share": 8.0,
    },

    # ---------------------------------------------------------------------
    # Scenario 4: China Soybean Trade Tensions (India Soybean Export)
    # ---------------------------------------------------------------------
    "china_soybean_trade": {
        "description": "Bilateral trade tension and tariff review on soybean shipments with China",
        "event_date": "2024-06-15",
        "event_country": "CHINA",
        "event_root": "16",
        "event_code": "163",
        "goldstein_score": -5.5,
        "avg_tone": -4.0,
        "num_mentions": 95,
        "commodity": "Soybean",
        "hs4": 1201,
        "trade_type": "Export",
        "shock_intensity": 1.5,
        "trade_share": 15.0,
    },

    # ---------------------------------------------------------------------
    # Scenario 5: Red Sea Shipping & Freight Disruption (UAE Agricultural Export)
    # ---------------------------------------------------------------------
    "red_sea_shipping_disruption": {
        "description": "Maritime security threat and freight rate surge along Red Sea / Gulf shipping routes",
        "event_date": "2024-06-15",
        "event_country": "UNITED ARAB EMIRATES",
        "event_root": "18",
        "event_code": "180",
        "goldstein_score": -7.0,
        "avg_tone": -5.2,
        "num_mentions": 110,
        "commodity": "Pepper",
        "hs4": 904,
        "trade_type": "Export",
        "shock_intensity": 2.0,
        "trade_share": 10.0,
    },
}

# Default scenario to run if none is specified via CLI
ACTIVE_SCENARIO = "wheat_russia_conflict"


def list_scenarios():
    """Print available scenarios."""
    print("\nAvailable Drishti Scenarios:")
    print("=" * 70)
    for key, sc in SCENARIOS.items():
        active_mark = " [*ACTIVE*]" if key == ACTIVE_SCENARIO else ""
        print(f"  * {key:<28}{active_mark}")
        print(f"    Description: {sc.get('description', 'N/A')}")
        print(f"    Partner:     {sc.get('event_country')} | Commodity: {sc.get('commodity')} (HS4: {sc.get('hs4')}) | Flow: {sc.get('trade_type')}")
        print()
    print("=" * 70)


def run_scenario(scenario_name: str):
    """Execute scenario through the Drishti Orchestrator."""
    if scenario_name not in SCENARIOS:
        print(f"[ERROR] Scenario '{scenario_name}' not found in SCENARIOS.")
        print(f"Available options: {list(SCENARIOS.keys())}")
        sys.exit(1)

    scenario_data = SCENARIOS[scenario_name]
    print(f"\n[Drishti] Loading Scenario: '{scenario_name}'")
    print(f"  Description: {scenario_data.get('description')}")
    print(f"  Event Date:  {scenario_data.get('event_date')} | Country: {scenario_data.get('event_country')}")
    print(f"  Commodity:   {scenario_data.get('commodity')} (HS4: {scenario_data.get('hs4')}) | Flow: {scenario_data.get('trade_type')}\n")

    # Import orchestrator lazily
    from agents.orchestrator import DrishtiOrchestrator
    orchestrator = DrishtiOrchestrator()
    orchestrator.run_scenario_dict(scenario_name, scenario_data)


def main():
    parser = argparse.ArgumentParser(description="Drishti Phase-1 Scenario Runner")
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default=ACTIVE_SCENARIO,
        help=f"Scenario key to execute (default: '{ACTIVE_SCENARIO}')"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all defined scenarios and exit"
    )

    args = parser.parse_args()

    if args.list:
        list_scenarios()
        return

    run_scenario(args.scenario)


if __name__ == "__main__":
    main()
