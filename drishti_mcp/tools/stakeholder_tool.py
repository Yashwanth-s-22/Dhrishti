"""
Drishti MCP Tool: Stakeholder Disaggregation Engine
===================================================
Interfaces with the deterministic rules-based stakeholder disaggregation engine
(scripts/stakeholder_engine.py). Computes structured effects and generates stakeholder
advisories across farmers, consumers, exporters, importers, regional risk, and government.
Preserves transparent rule-based logic and enforces canonical shock-direction consistency.
"""

from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.stakeholder_engine import (
    compute_stakeholder_effects,
    generate_advisory,
    RULE_BASED_LIMITATIONS,
)


def get_stakeholder_analysis(
    country: str,
    trade_type: str,
    hs4: int,
    commodity_name: str = "",
    trade_pred: Optional[float] = None,
    prod_growth: Optional[float] = None,
    prod_risk: str = "Medium",
    price_pred: Optional[float] = None,
    gva_pred: Optional[float] = None,
    infl_pred: Optional[float] = None,
    trade_share: float = 5.0,
    effective_shock: float = 0.0,
    canonical_shock_direction: str = "supply_contraction",
    month: int = 1,
    year: int = 2024,
) -> Dict[str, Any]:
    """
    Compute structured stakeholder impacts and advisories from quantitative cascade predictions.

    Args:
        country: Partner country
        trade_type: Trade position ('Export' or 'Import')
        hs4: 4-digit HS code
        commodity_name: Human-readable commodity name (e.g. 'Wheat')
        trade_pred: Predicted 1-month trade return % (Model A)
        prod_growth: Predicted national production growth % (Model B)
        prod_risk: Production risk tier ('Low', 'Medium', 'High', 'Critical')
        price_pred: Predicted 1-month price return % (Model C)
        gva_pred: Predicted Agri GVA growth % (Model D)
        infl_pred: Predicted 3-month inflation delta pp (Model D)
        trade_share: Partner trade share %
        effective_shock: Calculated effective shock exposure
        canonical_shock_direction: Canonical event classification shock direction
        month: Calendar month (1-12)
        year: Observation year

    Returns:
        Structured dictionary containing stakeholder effects, human-readable advisories, and consistency metrics.
    """
    cascade_state = {
        "trade": {"Trade_Return_1M_Pred": trade_pred},
        "agriculture": {
            "Production_Growth_Pred": prod_growth,
            "Production_Risk": prod_risk,
        },
        "price": {"Price_Return_1M_Pred": price_pred},
        "economy": {
            "Agri_GVA_Growth_Pred": gva_pred,
            "Inflation_Change_Pred": infl_pred,
        },
    }

    row_context = {
        "Country": country.strip().upper(),
        "Trade_Type": trade_type.strip().capitalize(),
        "HS4": int(hs4),
        "Commodity": commodity_name or f"HS4={hs4}",
        "Trade_Share": float(trade_share / 100.0) if trade_share > 1.0 else float(trade_share),
        "Effective_Shock": float(effective_shock),
        "Incoming_Shock_Exposure": float(effective_shock) if trade_type.lower() == "import" else 0.0,
        "Outgoing_Shock_Exposure": float(effective_shock) if trade_type.lower() == "export" else 0.0,
        "Month": int(month),
        "Year": int(year),
    }

    effects = compute_stakeholder_effects(cascade_state, row_context)
    advisory = generate_advisory(effects, row_context)

    derived_direction = effects.get("macro_summary", {}).get("shock_direction", "unknown")
    
    # Check consistency between canonical event direction and derived stakeholder heuristic
    is_consistent = (
        canonical_shock_direction.lower() in derived_direction.lower() or
        derived_direction.lower() in canonical_shock_direction.lower() or
        ("supply" in canonical_shock_direction.lower() and "supply" in derived_direction.lower()) or
        ("demand" in canonical_shock_direction.lower() and "demand" in derived_direction.lower())
    )

    consistency_status = "CONSISTENT" if is_consistent else "WARNING — interpretation differs from event classification"

    return {
        "status": "success",
        "context": advisory.get("context", ""),
        "canonical_shock_direction": canonical_shock_direction,
        "derived_shock_direction": derived_direction,
        "shock_direction_consistency": consistency_status,
        "effects": effects,
        "stakeholder_advisories": advisory.get("stakeholder_advisories", {}),
        "limitations": RULE_BASED_LIMITATIONS,
        "provenance": "[RULE-BASED OUTPUT]",
    }
