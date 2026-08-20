"""
Drishti - Task 8: Stakeholder Disaggregation Rules Engine
==========================================================
Rules-based layer (not ML) producing separate effects for:
- Farmer (output-price effect + input-cost effect)
- Consumer
- Exporter
- Importer
- Regional production risk

Logic is gated on trade position, partner dependency, production risk, and season.

Run: python scripts/stakeholder_engine.py

Design Notes and Methodological Clarifications
-----------------------------------------------
1. RULE-BASED ARCHITECTURE:
   This module is a deterministic rules-based decision layer, not an ML model.
   It translates numerical predictions from Models A-D into structured,
   disaggregated stakeholder impacts.

2. TRADE POSITION (FLOW DIRECTION):
   Trade_Type indicates the observed bilateral trade flow for this record:
   - 'Export': India is the exporter for this transaction.
   - 'Import': India is the importer for this transaction.
   This denotes flow direction for the specific commodity-partner pair, NOT
   India's overall national net trade balance.

3. SHOCK DIRECTION INTERPRETATION:
   determine_shock_direction() provides a directional interpretation of how
   predicted trade/price pressures propagate (e.g., supply shock vs demand shock).
   It does not claim causal event detection unless explicit event features support it.

4. FARMER OUTPUT-PRICE EFFECT VS INPUT COSTS:
   - Output-price effect captures the producer revenue implication of predicted price changes.
   - Input-cost effect is explicitly marked as a data gap because fertilizer/input trade
     data (HS Chapter 31) is absent in the dataset.

5. CONSUMER EFFECT & HEURISTICS:
   Price increases harm consumers, while price decreases benefit them.
   Production risk amplification is a project-defined heuristic, not econometric welfare measurement.

6. PROJECT-DEFINED THRESHOLDS:
   Trade_Share dependency tiers (>=30%, 10-30%, 1-10%, <1%) and shock triggers
   (e.g., Effective_Shock > 100) are project-defined heuristics.

7. REGIONAL PRODUCTION RISK:
   Phrased as 'Regional production risk' / 'Producing regions may face...'.
   Specific state/district claims are avoided because trade records lack sub-national disaggregation.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ============================================================
# DIRECTIONAL LOGIC & LIMITATION DOCUMENTATION
# ============================================================
DIRECTIONAL_RULES = """
STAKEHOLDER DISAGGREGATION LOGIC (Rules-Based Layer, Not ML-Trained)
-------------------------------------------------------------------
1. Observed Flow: India as Importer (Import Flow)
   - Disruption / contraction in import supply:
     * Farmer (output-price): Positive output-price effect if domestic price rises.
     * Farmer (input-cost):   Data gap (HS Ch. 31 fertilizer data absent).
     * Consumer:              Negative effect (higher domestic food prices).
     * Importer:              Negative effect (supply chain tightness / higher acquisition cost).
     * Exporter:              Not applicable (India is importer for this flow).
     * Regional:              Producing regions benefit from price realization; non-producing face price pressure.

2. Observed Flow: India as Exporter (Export Flow)
   - Contraction in export market access / demand shock:
     * Farmer (output-price): Negative output-price effect if export glut depresses prices.
     * Farmer (input-cost):   Data gap (HS Ch. 31 fertilizer data absent).
     * Consumer:              Positive effect if domestic availability increases and price softens.
     * Exporter:              Negative effect (revenue loss, disrupted buyer relationships).
     * Importer:              Not applicable (India is exporter for this flow).
     * Regional:              Export-oriented producing regions face output realization risks.

3. Partner Dependency & Shock Scaling:
   - Severity is scaled by partner Trade_Share using project-defined heuristic thresholds:
     * High (>= 30%), Moderate (10-30%), Low (1-10%), Negligible (< 1%).
"""

RULE_BASED_LIMITATIONS = {
    "framework": "Deterministic rules-based decision layer (not an ML model).",
    "purpose": "Interprets predicted trade/price/agricultural/economic impacts from upstream Models A-D.",
    "causal_attribution": "Does not establish causal attribution to specific geopolitical events unless event parameters are explicitly provided.",
    "thresholds": "Dependency tiers and shock triggers are project-defined heuristic thresholds.",
    "data_gaps": "Regional disaggregation is limited to national production risk proxies; fertilizer and input-cost impacts cannot be assessed due to absence of HS Chapter 31 data."
}


def classify_trade_position(trade_type):
    """
    Identify observed trade flow direction for this transaction:
    - 'Export': India is exporter for this observation
    - 'Import': India is importer for this observation
    Note: Denotes flow direction for the record, NOT national net exporter/importer status.
    """
    if trade_type == "Export":
        return "exporter"
    elif trade_type == "Import":
        return "importer"
    return "unknown"


def assess_dependency_level(trade_share):
    """
    Categorize partner dependency based on Trade_Share using project-defined rules:
    - >= 30%: High dependency (project-defined threshold)
    - 10% - 30%: Moderate dependency (project-defined threshold)
    - 1% - 10%: Low dependency (project-defined threshold)
    - < 1%: Negligible dependency (project-defined threshold)
    Note: These thresholds are heuristic classifications, not universally validated economic boundaries.
    """
    if trade_share >= 0.3:
        return "high"
    elif trade_share >= 0.1:
        return "moderate"
    elif trade_share >= 0.01:
        return "low"
    else:
        return "negligible"


def determine_shock_direction(trade_type, effective_shock=0.0, incoming_shock=0.0, outgoing_shock=0.0, price_pred=0.0, trade_pred=0.0):
    """
    Directional interpretation of shock transmission based on observed flow and model predictions.

    Distinguishes:
    - 'supply_shock': Import flow with predicted price rise / trade contraction or elevated shock exposure
    - 'demand_shock': Export flow with predicted trade decline / price drop or elevated foreign shock
    - 'export_flow': Standard export flow without acute disruption signature
    - 'import_flow': Standard import flow without acute disruption signature
    - 'neutral/unknown': No notable shock or indeterminate trade flow

    Note: This is a directional interpretation of predicted trade/price impacts,
    NOT a confirmed causal identification of a specific geopolitical event.
    """
    is_export = (trade_type == "Export")
    is_import = (trade_type == "Import")
    has_shock_exposure = (effective_shock > 0.05) or (incoming_shock > 0.01) or (outgoing_shock > 0.01)

    if is_import:
        if (price_pred > 0 and trade_pred < 0) or has_shock_exposure or (trade_pred < -0.05):
            return "supply_shock"
        return "import_flow"
    elif is_export:
        if (trade_pred < 0) or (price_pred < 0) or has_shock_exposure:
            return "demand_shock"
        return "export_flow"
    else:
        return "neutral/unknown"


def compute_stakeholder_effects(cascade_state, row):
    """
    Compute disaggregated stakeholder effects from cascade state.

    Args:
        cascade_state: dict with trade/agriculture/price/economy predictions
        row: dict with feature values for this observation

    Returns:
        dict of structured stakeholder effects
    """
    # Extract model predictions (safely handling None when upstream stage is unavailable)
    trade_raw = cascade_state.get("trade", {}).get("Trade_Return_1M_Pred", None)
    trade_pred = float(trade_raw) if trade_raw is not None else None

    prod_raw = cascade_state.get("agriculture", {}).get("Production_Growth_Pred", None)
    prod_growth = float(prod_raw) if prod_raw is not None else None
    prod_risk = str(cascade_state.get("agriculture", {}).get("Production_Risk", "Unknown"))

    price_raw = cascade_state.get("price", {}).get("Price_Return_1M_Pred", None)
    price_pred = float(price_raw) if price_raw is not None else None

    gva_raw = cascade_state.get("economy", {}).get("Agri_GVA_Growth_Pred", None)
    gva_pred = float(gva_raw) if gva_raw is not None else None

    infl_raw = cascade_state.get("economy", {}).get("Inflation_Change_Pred", None)
    infl_pred = float(infl_raw) if infl_raw is not None else None

    # Extract context
    trade_type = row.get("Trade_Type", "Unknown")
    trade_share = float(row.get("Trade_Share", 0.0))
    effective_shock = float(row.get("Effective_Shock", 0.0))
    incoming = float(row.get("Incoming_Shock_Exposure", 0.0))
    outgoing = float(row.get("Outgoing_Shock_Exposure", 0.0))
    month = int(row.get("Month", 1))

    # Classify context using project-defined rules
    position = classify_trade_position(trade_type)
    dependency = assess_dependency_level(trade_share)
    shock_dir = determine_shock_direction(
        trade_type=trade_type,
        effective_shock=effective_shock,
        incoming_shock=incoming,
        outgoing_shock=outgoing,
        price_pred=price_pred if price_pred is not None else 0.0,
        trade_pred=trade_pred if trade_pred is not None else 0.0,
    )

    # Determine seasonal calendar context
    if month in [6, 7, 8, 9, 10]:
        season = "Kharif"
    elif month in [11, 12, 1, 2, 3]:
        season = "Rabi"
    else:
        season = "Summer/Zaid"

    effects = {}

    # Severity multiplier based on project-defined dependency tier
    dep_mult = {"high": 1.5, "moderate": 1.0, "low": 0.5, "negligible": 0.1}[dependency]

    # ----------------------------------------------------------
    # 1. Farmer Output-Price Effect
    # ----------------------------------------------------------
    if price_pred is not None:
        farmer_output_magnitude = price_pred * dep_mult
        if price_pred > 0:
            farmer_output_dir = "positive"
            if shock_dir == "supply_shock":
                farmer_output_reason = "Positive output-price effect: import contraction / upward price pressure improves price realization for domestic producers"
            else:
                farmer_output_reason = "Positive output-price effect: higher predicted commodity price improves realization for domestic producers"
        elif price_pred < 0:
            farmer_output_dir = "negative"
            if shock_dir == "demand_shock":
                farmer_output_reason = "Negative output-price effect: export contraction / domestic oversupply pressures farmgate realization downward"
            else:
                farmer_output_reason = "Negative output-price effect: lower predicted commodity price reduces realization for domestic producers"
        else:
            farmer_output_dir = "neutral"
            farmer_output_reason = "Neutral output-price effect: no significant commodity price change predicted"
    else:
        farmer_output_magnitude = 0.0
        farmer_output_dir = "unavailable"
        farmer_output_reason = "Cannot assess output-price effect: upstream Model C price prediction unavailable (Model B Lag1 missing/non-crop)"

    effects["farmer_output_price"] = {
        "effect": float(farmer_output_magnitude),
        "direction": farmer_output_dir,
        "reason": farmer_output_reason,
        "severity": dependency,
        "relevant_input": {
            "price_return_1m_pred": price_pred,
            "shock_direction": shock_dir,
            "dependency_level": dependency,
        },
        "data_gap": price_pred is None,
    }

    # ----------------------------------------------------------
    # 2. Farmer Input-Cost Effect (Documented Data Gap)
    # ----------------------------------------------------------
    # No fertilizer / input-cost trade data in dataset (HS Chapter 31 absent)
    effects["farmer_input_cost"] = {
        "effect": 0.0,
        "direction": "not_assessed",
        "reason": "Cannot assess: fertilizer and agricultural input-cost trade data absent (HS Chapter 31 absent in dataset)",
        "severity": "unknown",
        "relevant_input": {
            "hs_chapter_31_available": False,
        },
        "data_gap": True,
    }

    # ----------------------------------------------------------
    # 3. Consumer Effect
    # ----------------------------------------------------------
    if price_pred is not None:
        consumer_magnitude = -price_pred * dep_mult
        if price_pred > 0:
            consumer_dir = "negative"
            consumer_reason = "Negative consumer impact: higher predicted commodity prices increase consumer food expenditure"
        elif price_pred < 0:
            consumer_dir = "positive"
            consumer_reason = "Positive consumer impact: lower predicted commodity prices reduce consumer food expenditure"
        else:
            consumer_dir = "neutral"
            consumer_reason = "Neutral consumer impact: no significant commodity price change predicted"

        # Directional heuristic: amplify consumer risk if production risk is elevated
        if prod_risk in ["Critical", "High"] and price_pred > 0:
            consumer_magnitude *= 1.3
            consumer_reason += " (amplified by elevated regional production risk; project-defined directional heuristic)"
    else:
        consumer_magnitude = 0.0
        consumer_dir = "unavailable"
        consumer_reason = "Cannot assess consumer impact: upstream Model C price prediction unavailable (Model B Lag1 missing/non-crop)"

    effects["consumer"] = {
        "effect": float(consumer_magnitude),
        "direction": consumer_dir,
        "reason": consumer_reason,
        "severity": dependency,
        "relevant_input": {
            "price_return_1m_pred": price_pred,
            "production_risk": prod_risk,
        },
        "data_gap": price_pred is None,
    }

    # ----------------------------------------------------------
    # 4. Exporter Effect
    # ----------------------------------------------------------
    if position == "exporter":
        exporter_magnitude = trade_pred * dep_mult
        if trade_pred > 0:
            exporter_dir = "positive"
            exporter_reason = "Positive exporter impact: predicted export volume expansion indicates favorable market access and revenue potential"
        elif trade_pred < 0:
            exporter_dir = "negative"
            exporter_reason = "Negative exporter impact: predicted export volume contraction indicates market contraction and revenue reduction"
        else:
            exporter_dir = "neutral"
            exporter_reason = "Neutral exporter impact: no significant export volume change predicted"
        exporter_severity = dependency
    else:
        exporter_magnitude = 0.0
        exporter_dir = "not_applicable"
        exporter_reason = "Not applicable: observed flow is Import (India is importer, not exporter, for this record)"
        exporter_severity = "not_applicable"

    effects["exporter"] = {
        "effect": float(exporter_magnitude),
        "direction": exporter_dir,
        "reason": exporter_reason,
        "severity": exporter_severity,
        "relevant_input": {
            "trade_return_1m_pred": float(trade_pred),
            "trade_position": position,
        },
        "data_gap": False,
    }

    # ----------------------------------------------------------
    # 5. Importer Effect
    # ----------------------------------------------------------
    if position == "importer":
        # Check project-defined threshold for severe shock
        if effective_shock > 100:  # Project-defined threshold
            importer_magnitude = -abs(trade_pred if trade_pred != 0 else 1.0) * dep_mult
            importer_dir = "negative"
            importer_reason = "Negative importer impact: elevated shock exposure (Effective_Shock > 100, project-defined threshold) threatens import supply continuity"
        elif trade_pred < 0:
            importer_magnitude = trade_pred * dep_mult
            importer_dir = "negative"
            importer_reason = "Negative importer impact: predicted import volume contraction signals potential supply chain tightness"
        elif trade_pred > 0:
            importer_magnitude = trade_pred * dep_mult * 0.5
            importer_dir = "positive"
            importer_reason = "Positive importer impact: expanding import volume supports domestic supply availability"
        else:
            importer_magnitude = 0.0
            importer_dir = "neutral"
            importer_reason = "Neutral importer impact: import flow relatively stable without acute disruption signals"
        importer_severity = dependency
    else:
        importer_magnitude = 0.0
        importer_dir = "not_applicable"
        importer_reason = "Not applicable: observed flow is Export (India is exporter, not importer, for this record)"
        importer_severity = "not_applicable"

    effects["importer"] = {
        "effect": float(importer_magnitude),
        "direction": importer_dir,
        "reason": importer_reason,
        "severity": importer_severity,
        "relevant_input": {
            "trade_return_1m_pred": trade_pred,
            "effective_shock": float(effective_shock),
            "trade_position": position,
        },
        "data_gap": trade_pred is None,
    }

    # ----------------------------------------------------------
    # 6. Regional Production Risk
    # ----------------------------------------------------------
    # Phrased as regional risk proxy without fabricating sub-national disaggregation
    if prod_risk in ["Critical", "High"]:
        regional_dir = "negative"
        regional_reason = f"Elevated production risk ({prod_risk}): producing regions may face output shortfalls (note: district/state disaggregation not present in trade record)"
    elif prod_risk == "Medium":
        regional_dir = "caution"
        regional_reason = "Moderate production risk: producing regions should monitor seasonal trends"
    else:
        regional_dir = "stable"
        regional_reason = "Low production risk: producing regions show stable baseline outlook"

    effects["regional_production_risk"] = {
        "risk_level": prod_risk,
        "direction": regional_dir,
        "reason": regional_reason,
        "season": season,
        "production_growth": float(prod_growth) if prod_growth is not None else None,
        "relevant_input": {
            "production_growth_pred": prod_growth,
            "production_risk": prod_risk,
            "season": season,
        },
        "data_gap": prod_growth is None,
    }

    # ----------------------------------------------------------
    # 7. Macro Summary
    # ----------------------------------------------------------
    effects["macro_summary"] = {
        "gva_growth_pred": float(gva_pred) if gva_pred is not None else None,
        "inflation_change_pred": float(infl_pred) if infl_pred is not None else None,
        "trade_return_pred": float(trade_pred) if trade_pred is not None else None,
        "price_return_pred": float(price_pred) if price_pred is not None else None,
        "shock_direction": shock_dir,
        "trade_position": position,
        "dependency_level": dependency,
    }

    return effects


def generate_advisory(effects, row):
    """
    Generate human-readable advisory for each stakeholder from computed effects.
    """
    country = row.get("Country", "Unknown")
    commodity = row.get("Commodity", f"HS4={row.get('HS4', '?')}")
    trade_type = row.get("Trade_Type", "?")
    year = int(row.get("Year", 0))
    month = int(row.get("Month", 0))
    year_month = f"{year}-{month:02d}"

    advisory = {
        "context": f"{trade_type} flow of {commodity} with {country} ({year_month})",
        "methodology_note": "Rules-based interpretation of Model A-D cascade predictions (deterministic decision layer).",
        "limitations": RULE_BASED_LIMITATIONS,
        "stakeholder_advisories": {},
    }

    # Farmer advisory
    fp = effects["farmer_output_price"]
    fic = effects["farmer_input_cost"]
    advisory["stakeholder_advisories"]["farmers"] = (
        f"Output-price effect: {fp['direction'].upper()}. {fp['reason']}. "
        f"Partner dependency: {fp['severity']} (project-defined rule). "
        f"Input-cost effect: {fic['reason']}."
    )

    # Consumer advisory
    cons = effects["consumer"]
    advisory["stakeholder_advisories"]["consumers"] = (
        f"Consumer impact: {cons['direction'].upper()}. {cons['reason']}. "
        f"Severity tier: {cons['severity']}."
    )

    # Exporter advisory
    ex = effects["exporter"]
    advisory["stakeholder_advisories"]["exporters"] = (
        f"Exporter impact: {ex['direction'].upper()}. {ex['reason']}."
    )

    # Importer advisory
    im = effects["importer"]
    advisory["stakeholder_advisories"]["importers"] = (
        f"Importer impact: {im['direction'].upper()}. {im['reason']}."
    )

    # Regional advisory
    rp = effects["regional_production_risk"]
    prod_growth_text = f"Projected national production growth: {rp['production_growth']:+.1f}%." if rp['production_growth'] is not None else "Production growth: N/A."
    advisory["stakeholder_advisories"]["regional"] = (
        f"Regional risk: {rp['risk_level']}. {rp['reason']}. "
        f"Season: {rp['season']}. {prod_growth_text}"
    )

    # Government / Macro advisory
    macro = effects["macro_summary"]
    gva_txt = f"{macro['gva_growth_pred']:+.2f}%" if macro['gva_growth_pred'] is not None else "N/A"
    infl_txt = f"{macro['inflation_change_pred']:+.2f} pp" if macro['inflation_change_pred'] is not None else "N/A"
    trade_txt = f"{macro['trade_return_pred']:+.2f}%" if macro['trade_return_pred'] is not None else "N/A"
    price_txt = f"{macro['price_return_pred']:+.2f}%" if macro['price_return_pred'] is not None else "N/A (Model B Lag1 missing)"
    advisory["stakeholder_advisories"]["government"] = (
        f"Macro outlook: Agri GVA growth pred = {gva_txt}, "
        f"Inflation change pred = {infl_txt}. "
        f"Trade return pred = {trade_txt}, "
        f"Price return pred = {price_txt}. "
        f"Interpreted shock direction: {macro['shock_direction']}."
    )

    return advisory


def main():
    print("=" * 70)
    print("Drishti - Task 8: Stakeholder Disaggregation Engine")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("RULE-BASED INTERPRETATION LIMITATIONS & SCOPE")
    print("=" * 70)
    for key, text in RULE_BASED_LIMITATIONS.items():
        print(f"  - {key.replace('_', ' ').capitalize()}: {text}")

    print("\n" + "=" * 70)
    print("DIRECTIONAL LOGIC OVERVIEW")
    print("=" * 70)
    print(DIRECTIONAL_RULES)

    # Load cascade results
    cascade_path = os.path.join(RESULTS_DIR, "cascade_results.json")
    if not os.path.exists(cascade_path):
        print(f"\n  ERROR: Cascade results not found at {cascade_path}")
        print("  Run cascade_orchestrator.py first.")
        return

    with open(cascade_path) as f:
        cascade_results = json.load(f)

    print(f"\nProcessing {len(cascade_results)} cascade demonstration cases...")

    all_advisories = []
    for i, result in enumerate(cascade_results):
        meta = result["row_meta"]
        # Handle both cascade_predictions and legacy cascade_state key
        state = result.get("cascade_predictions") or result.get("cascade_state", {})
        row = {**meta, **state.get("exposure_features", {})}

        print("\n" + "-" * 70)
        print(f"Case {i+1}: {meta['Country']} | Observed Flow: {meta.get('Trade_Type', '?')} | HS4={meta['HS4']} | {meta['Year']}-{meta['Month']:02d}")
        print("-" * 70)

        effects = compute_stakeholder_effects(state, row)
        advisory = generate_advisory(effects, row)

        # Print structured advisory output
        for stakeholder, text in advisory["stakeholder_advisories"].items():
            print(f"  [{stakeholder.upper():<10}] {text}")

        all_advisories.append({
            "meta": meta,
            "effects": effects,
            "advisory": advisory,
        })

    # Save output
    output_path = os.path.join(RESULTS_DIR, "stakeholder_advisories.json")
    with open(output_path, "w") as f:
        json.dump(all_advisories, f, indent=2, default=str)
    print(f"\nAdvisories saved: {output_path}")

    print("\n" + "=" * 70)
    print("TASK 8 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
