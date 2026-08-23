"""
Drishti MCP Tool: Historical Event Store
========================================
Queries the curated Drishti Event Store (results/event_store.json and data/event_catalog.json)
to retrieve historical geopolitical shock events, matched commodity trade impacts,
and Model A/C performance benchmarks.
Preserves DIRECT vs PROXY event labeling, computed relevance rationale, and non-causal observational framing.
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, Any, List, Optional

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import EVENT_STORE_PATH, EVENT_CATALOG_PATH

_EVENT_STORE_CACHE = None
_EVENT_CATALOG_CACHE = None


def _load_event_data():
    """Load and cache event store and catalog JSON files."""
    global _EVENT_STORE_CACHE, _EVENT_CATALOG_CACHE
    if _EVENT_STORE_CACHE is None and os.path.exists(EVENT_STORE_PATH):
        with open(EVENT_STORE_PATH, "r", encoding="utf-8") as f:
            _EVENT_STORE_CACHE = json.load(f)

    if _EVENT_CATALOG_CACHE is None and os.path.exists(EVENT_CATALOG_PATH):
        with open(EVENT_CATALOG_PATH, "r", encoding="utf-8") as f:
            _EVENT_CATALOG_CACHE = json.load(f)


def _compute_relevance_rationale(ev_meta: Dict[str, Any], query: str, commodity: str, country: str) -> str:
    """Compute explicit, human-readable selection rationale for historical analog."""
    ev_countries = [c.upper() for c in ev_meta.get("countries", [])]
    ev_name = ev_meta.get("name", "")
    ev_desc = ev_meta.get("description", "")
    scope = ev_meta.get("event_scope", "direct")
    
    reasons = []
    if country and country.upper() in ev_countries:
        reasons.append(f"Same partner country ({country.upper()})")
    
    if commodity and (commodity.lower() in ev_name.lower() or commodity.lower() in ev_desc.lower()):
        reasons.append(f"Same target commodity ({commodity})")
    elif commodity and ("oil" in commodity.lower() and "oil" in ev_desc.lower()):
        reasons.append("Same commodity family (Edible Oils)")
    elif commodity and (commodity.lower() in ["wheat", "rice", "corn", "maize"] and any(g in ev_desc.lower() for g in ["grain", "wheat", "rice", "cereal"])):
        reasons.append("Same commodity family (Cereals & Foodgrains)")

    if "export" in ev_desc.lower() or "ban" in ev_desc.lower() or "restriction" in ev_desc.lower():
        reasons.append("Similar export restriction / trade barrier dynamics")
    elif "conflict" in ev_desc.lower() or "war" in ev_desc.lower():
        reasons.append("Geopolitical conflict & shipping route disruption")
    elif "freight" in ev_desc.lower() or "shipping" in ev_desc.lower():
        reasons.append("Maritime logistics & freight disruption")

    if scope == "proxy":
        reasons.append("Evaluated as geographic trade proxy")

    return "; ".join(reasons) if reasons else f"Topical historical analog for {commodity or country or query}"


def get_historical_event(
    event_id: str = "",
    query: str = "",
    commodity: str = "",
    country: str = "",
) -> Dict[str, Any]:
    """
    Search and retrieve historical event analysis from the Drishti Event Store.

    Args:
        event_id: Specific event identifier (e.g. 'EVT001', 'EVT002', 'EVT006')
        query: Free-text keyword search across event name, description, and narrative
        commodity: Commodity keyword (e.g. 'wheat', 'rice', 'palm oil')
        country: Country name (e.g. 'Russia', 'Ukraine', 'Bangladesh')

    Returns:
        Structured dictionary with matched events, observational window statistics, selection relevance, and provenance.
    """
    _load_event_data()

    if not _EVENT_STORE_CACHE:
        return {
            "status": "unavailable",
            "message": "Historical event store artifact is not available.",
            "events": [],
            "provenance": "[HISTORICAL EVENT STORE]",
        }

    matched_events = []
    e_id_clean = event_id.strip().upper()
    q_clean = query.strip().lower()
    comm_clean = commodity.strip().lower()
    ctry_clean = country.strip().lower()

    for item in _EVENT_STORE_CACHE:
        ev_meta = item.get("event", {})
        ev_id = ev_meta.get("event_id", "")
        ev_name = ev_meta.get("name", "")
        ev_desc = ev_meta.get("description", "")
        ev_notes = ev_meta.get("notes", "")
        ev_countries = [c.lower() for c in ev_meta.get("countries", [])]
        ev_hs4 = [str(c).lower() for c in ev_meta.get("hs4_codes", []) + ev_meta.get("hs4_commodities", [])]
        
        searchable_text = f"{ev_name} {ev_desc} {ev_notes}".lower()

        is_match = False

        if e_id_clean and e_id_clean == ev_id:
            is_match = True
        elif q_clean and (q_clean in searchable_text or any(q_clean in c for c in ev_countries)):
            is_match = True
        elif comm_clean and (comm_clean in searchable_text or any(comm_clean in c for c in ev_hs4)):
            is_match = True
        elif ctry_clean and any(ctry_clean in c for c in ev_countries):
            is_match = True

        if is_match or (not e_id_clean and not q_clean and not comm_clean and not ctry_clean):
            summary = item.get("summary", {})
            obs_metrics = summary.get("observed_metrics_window", {})
            preds = item.get("predictions", {})

            relevance = _compute_relevance_rationale(ev_meta, q_clean, commodity, country)

            matched_events.append({
                "event_id": ev_id,
                "name": ev_name,
                "description": ev_desc,
                "event_scope": ev_meta.get("event_scope", "direct"),
                "relevance": relevance,
                "start_date": ev_meta.get("start") or ev_meta.get("start_date"),
                "end_date": ev_meta.get("end") or ev_meta.get("end_date"),
                "severity": ev_meta.get("severity"),
                "countries": ev_meta.get("countries", []),
                "hs4_codes": ev_meta.get("hs4_codes", []),
                "total_matched_rows": summary.get("n_rows", 0),
                "observed_trade_impact": {
                    "avg_trade_return_1m": obs_metrics.get("avg_observed_trade_return_1m"),
                    "avg_trade_value_usd": obs_metrics.get("avg_trade_value_usd"),
                    "total_trade_value_usd": obs_metrics.get("total_trade_value_usd"),
                },
                "observed_price_impact": {
                    "avg_price_return_1m": obs_metrics.get("avg_observed_price_return_1m"),
                },
                "model_predictions_vs_actual": {
                    "model_a": preds.get("trade_pred_vs_actual", {}),
                    "model_c": preds.get("price_pred_vs_actual", {}),
                },
                "coverage_metrics": preds.get("coverage_metrics", {}),
                "narrative": ev_notes or ev_desc,
            })

    return {
        "status": "success" if matched_events else "not_found",
        "matched_count": len(matched_events),
        "events": matched_events,
        "methodological_note": (
            "Historical event-window statistics describe observed trade and price movements during curated shock periods. "
            "They represent observational associations and do not claim unconfounded econometric causal identification."
        ),
        "provenance": "[HISTORICAL EVENT STORE]",
    }
