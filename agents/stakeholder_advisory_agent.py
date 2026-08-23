"""
Drishti Agent: Stakeholder Advisory Agent
=========================================
Interfaces with the deterministic stakeholder disaggregation engine via MCP to evaluate
distributional impacts across key agricultural economy participants:
- Farmers (Output price realization vs Input cost data gap)
- Consumers (Food expenditure impact)
- Exporters (Market access & revenue)
- Importers (Procurement tightness & continuity)
- Regional stakeholders (Producing vs consuming zones)
- Government (Macro balances & market intervention)

Preserves transparent deterministic rule scoring and enforces canonical shock-direction consistency.
"""

from typing import Dict, Any, Optional
from drishti_mcp.drishti_mcp_server import call_mcp_tool
from llm.gemini_client import GeminiClient


class StakeholderAdvisoryAgent:
    """
    Agent responsible for deterministic stakeholder disaggregation and advisory structuring.
    """

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()

    def process(
        self,
        event_dict: Dict[str, Any],
        ml_predictions: Dict[str, Any],
        trade_share: float = 5.0,
        effective_shock: float = 0.0,
        month: int = 1,
        year: int = 2024,
    ) -> Dict[str, Any]:
        """
        Compute stakeholder effects via MCP and structure domain advisories.

        Args:
            event_dict: Event parameters dictionary
            ml_predictions: ML cascade predictions dictionary
            trade_share: Partner trade share percentage
            effective_shock: Calculated effective shock exposure
            month: Analysis month
            year: Analysis year

        Returns:
            Dictionary containing structured stakeholder effects, driver breakdowns, and advisories.
        """
        country = event_dict.get("country", "RUSSIA")
        trade_type = event_dict.get("trade_type", "Export")
        hs4 = int(event_dict.get("hs4", 1001))
        commodity = event_dict.get("commodity", "Wheat")
        canonical_dir = event_dict.get("shock_direction", "supply_contraction")

        trade_pred = ml_predictions.get("trade", {}).get("Trade_Return_1M_Pred")
        prod_growth = ml_predictions.get("agriculture", {}).get("Production_Growth_Pred")
        prod_risk = ml_predictions.get("agriculture", {}).get("Production_Risk", "Medium")
        price_pred = ml_predictions.get("price", {}).get("Price_Return_1M_Pred")
        gva_pred = ml_predictions.get("economy", {}).get("Agri_GVA_Growth_Pred")
        infl_pred = ml_predictions.get("economy", {}).get("Inflation_Change_Pred")

        stakeholder_response = call_mcp_tool("get_stakeholder_analysis", {
            "country": country,
            "trade_type": trade_type,
            "hs4": hs4,
            "commodity_name": commodity,
            "trade_pred": trade_pred,
            "prod_growth": prod_growth,
            "prod_risk": prod_risk,
            "price_pred": price_pred,
            "gva_pred": gva_pred,
            "infl_pred": infl_pred,
            "trade_share": trade_share,
            "effective_shock": effective_shock,
            "canonical_shock_direction": canonical_dir,
            "month": month,
            "year": year,
        })

        effects = stakeholder_response.get("effects", {})
        advisories = stakeholder_response.get("stakeholder_advisories", {})

        return {
            "stakeholder_impacts": {
                "farmers": {
                    "output_price_effect": effects.get("farmer_output_price", {}),
                    "input_cost_effect": effects.get("farmer_input_cost", {}),
                    "advisory": advisories.get("farmers", ""),
                    "provenance": "[RULE-BASED OUTPUT]",
                },
                "consumers": {
                    "effect": effects.get("consumer", {}),
                    "advisory": advisories.get("consumers", ""),
                    "provenance": "[RULE-BASED OUTPUT]",
                },
                "exporters": {
                    "effect": effects.get("exporter", {}),
                    "advisory": advisories.get("exporters", ""),
                    "provenance": "[RULE-BASED OUTPUT]",
                },
                "importers": {
                    "effect": effects.get("importer", {}),
                    "advisory": advisories.get("importers", ""),
                    "provenance": "[RULE-BASED OUTPUT]",
                },
                "regional": {
                    "effect": effects.get("regional_production_risk", {}),
                    "advisory": advisories.get("regional", ""),
                    "provenance": "[RULE-BASED OUTPUT]",
                },
                "government": {
                    "macro_summary": effects.get("macro_summary", {}),
                    "advisory": advisories.get("government", ""),
                    "provenance": "[RULE-BASED OUTPUT]",
                },
            },
            "canonical_shock_direction": stakeholder_response.get("canonical_shock_direction", canonical_dir),
            "derived_shock_direction": stakeholder_response.get("derived_shock_direction", ""),
            "shock_direction_consistency": stakeholder_response.get("shock_direction_consistency", "CONSISTENT"),
            "limitations": stakeholder_response.get("limitations", {}),
            "provenance": "[RULE-BASED OUTPUT]",
        }
