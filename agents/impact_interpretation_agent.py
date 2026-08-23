"""
Drishti Agent: Impact Interpretation Agent
==========================================
Executes the quantitative Drishti ML Cascade (Models A -> B -> C -> D) via MCP and translates
the resulting econometric predictions into clear, decision-grade economic interpretations.
STRICT RULES:
1. Preserves exact numerical values, signs, and units without alteration.
2. NEVER makes causal claims (statistical association and multi-stage forecasts, not causal attribution).
3. Uses proper trade flow terminology (India as baseline reference).
"""

from typing import Dict, Any, Optional
from drishti_mcp.drishti_mcp_server import call_mcp_tool
from llm.gemini_client import GeminiClient


class ImpactInterpretationAgent:
    """
    Agent responsible for quantitative cascade execution and non-causal economic narrative generation.
    """

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()

    def process(
        self,
        event_dict: Dict[str, Any],
        shock_intensity: float = 1.0,
        trade_share: float = 5.0,
        year: int = 2024,
        month: int = 1,
    ) -> Dict[str, Any]:
        """
        Execute ML cascade via MCP and synthesize economic interpretation.

        Args:
            event_dict: Structured event dictionary from Event Intelligence Agent
            shock_intensity: Shock intensity index
            trade_share: Partner trade share percentage
            year: Analysis year
            month: Analysis month

        Returns:
            Dictionary with exact quantitative ML predictions, observed ground truth comparison, and qualitative interpretation.
        """
        country = event_dict.get("country", "RUSSIA")
        trade_type = event_dict.get("trade_type", "Export")
        trade_flow_desc = event_dict.get("trade_flow_description") or f"India's {trade_type.lower()}s with {country}"
        hs4 = int(event_dict.get("hs4", 1001))
        commodity = event_dict.get("commodity", "Wheat")

        # 1. Execute ML Cascade via MCP
        cascade_response = call_mcp_tool("run_drishti_ml_cascade", {
            "country": country,
            "trade_type": trade_type,
            "hs4": hs4,
            "year": year,
            "month": month,
            "shock_intensity": shock_intensity,
            "trade_share": trade_share,
        })

        predictions = cascade_response.get("predictions", {})
        observed_metrics = cascade_response.get("observed_metrics", {})
        coverage_info = cascade_response.get("coverage", {})

        trade_pred = predictions.get("trade", {}).get("Trade_Return_1M_Pred")
        prod_growth_pred = predictions.get("agriculture", {}).get("Production_Growth_Pred")
        prod_risk = predictions.get("agriculture", {}).get("Production_Risk", "Medium")
        price_pred = predictions.get("price", {}).get("Price_Return_1M_Pred")
        price_status = predictions.get("price", {}).get("status", "UNAVAILABLE")
        price_reason = predictions.get("price", {}).get("unavailable_reason")
        gva_pred = predictions.get("economy", {}).get("Agri_GVA_Growth_Pred")
        infl_pred = predictions.get("economy", {}).get("Inflation_Change_Pred")

        # 2. Construct LLM prompt for economic interpretation
        trade_text = f"{trade_pred:+.4f}%" if trade_pred is not None else "N/A"
        prod_text = f"{prod_growth_pred:+.4f}%" if prod_growth_pred is not None else "N/A"
        price_text = f"{price_pred:+.2f}%" if price_pred is not None else f"Unavailable ({price_status}: {price_reason})"
        gva_text = f"{gva_pred:+.4f}%" if gva_pred is not None else "N/A"
        infl_text = f"{infl_pred:+.4f} percentage points" if infl_pred is not None else "N/A"

        prompt = f"""
Provide a concise economic interpretation of the following multi-stage econometric ML cascade predictions for {commodity} (HS4: {hs4}) in {trade_flow_desc}:

Quantitative ML Outputs:
1. Model A (Trade Flow Return 1M): {trade_text}
2. Model B (National Production Growth): {prod_text} | Production Risk Tier: {prod_risk}
3. Model C (Domestic Price Return 1M): {price_text}
4. Model D (Agricultural GVA Growth): {gva_text}
5. Model D (Food Inflation 3M Delta): {infl_text}

Cascade State: {coverage_info.get('cascade_state', 'PARTIAL')} ({coverage_info.get('coverage_ratio', '')} outputs available)

CRITICAL PUBLICATION-GRADE INSTRUCTIONS:
- You must reference the EXACT numbers above. Do not alter or round them differently in your key statements.
- NON-CAUSAL LANGUAGE: Do NOT write causal claims such as 'trade contraction caused production to drop' or 'import shocks cause inflation'.
  Instead use terms like: 'Model A forecasts a trade flow return of...', 'Model B independently estimates...', 'Model D indicators suggest...', 'Statistical associations indicate...'.
- TRADE DIRECTION RULE: The trade flow is '{trade_type}' ({trade_flow_desc}).
  * If Trade Flow is Export: Model A negative trade return represents export shipment contraction / weaker export flows to {country}.
  * If Trade Flow is Import: Model A negative trade return represents import arrival contraction / weaker import receipts from {country}.
- Explicitly note if Model C Price Return is unavailable because the previous-period agricultural crop lag is not present.
- Explain the trade transmission context strictly from India's perspective ({trade_flow_desc}).
- Structure the response with 3 concise, clear paragraphs:
  1. Trade & Cross-Border Transmission
  2. Agricultural Production & Commodity Price Realization
  3. Macroeconomic & Inflationary Implications
"""

        system_instruction = (
            "You are the Drishti Impact Interpretation Agent. Your duty is to explain the economic "
            "mechanisms behind the quantitative ML cascade predictions using rigorous, non-causal econometric "
            "terminology. Never alter the mathematical model outputs. Never claim causal attribution."
        )

        narrative = self.llm.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        return {
            "ml_predictions": {
                "trade": predictions.get("trade", {}),
                "agriculture": predictions.get("agriculture", {}),
                "price": predictions.get("price", {}),
                "economy": predictions.get("economy", {}),
                "cascade_state": coverage_info.get("cascade_state", "PARTIAL"),
                "coverage": coverage_info,
                "provenance": "[ML MODEL OUTPUT]",
            },
            "observed_metrics": observed_metrics,
            "economic_interpretation": {
                "summary": narrative,
                "methodological_framing": "Econometric association and forecasting; does not establish causal attribution.",
                "llm_provider": getattr(self.llm, "last_provider_used", None) or "Offline Synthesis",
                "provenance": "[LLM INFERENCE]",
            },
        }
