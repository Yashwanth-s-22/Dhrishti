"""
Drishti Agent: Mitigation & Action Agent
========================================
Synthesizes concrete, actionable mitigation options and policy resilience playbooks across:
- Government (Market operations, buffer stock deployment, tariff quota calibration)
- Farmers (Crop diversification, MSP access, storage financing)
- Consumers (Targeted PDS deployment, stabilized retail network access)
- Exporters (Destination diversification, forward contract hedging)
- Importers (Bilateral sourcing corridors, rolling inventory management)

Grounded strictly in:
1. Current verified event parameters
2. Quantitative ML cascade outputs
3. Structured stakeholder impact drivers
4. Historical event store benchmarks & relevance rationale (via MCP get_historical_event)

Strictly labels recommendations as exploratory decision support [LLM INFERENCE].
"""

from typing import Dict, Any, List, Optional
from drishti_mcp.drishti_mcp_server import call_mcp_tool
from llm.gemini_client import GeminiClient


class MitigationActionAgent:
    """
    Agent responsible for synthesizing stakeholder-specific mitigation strategies
    informed by historical precedents and quantitative shock severity.
    """

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()

    def process(
        self,
        event_dict: Dict[str, Any],
        ml_predictions: Dict[str, Any],
        stakeholder_impacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synthesize multi-stakeholder mitigation options grounded in ML outputs and historical analogs.

        Args:
            event_dict: Event parameters dictionary
            ml_predictions: ML cascade outputs dictionary
            stakeholder_impacts: Stakeholder impacts dictionary

        Returns:
            Dictionary containing historical context with selection rationale and structured mitigation actions.
        """
        commodity = event_dict.get("commodity", "Wheat")
        country = event_dict.get("country", "RUSSIA")
        trade_type = event_dict.get("trade_type", "Export")
        trade_flow_desc = event_dict.get("trade_flow_description") or f"India's {trade_type.lower()}s with {country}"
        event_type = event_dict.get("event_type", "supply_shock")
        canonical_dir = event_dict.get("shock_direction", "supply_contraction")

        # 1. Query historical analogs from event store via MCP
        hist_response = call_mcp_tool("get_historical_event", {
            "commodity": commodity,
            "country": country,
        })
        historical_events = hist_response.get("events", [])

        # If no direct match, query general events
        if not historical_events:
            gen_hist = call_mcp_tool("get_historical_event", {"query": commodity})
            historical_events = gen_hist.get("events", [])

        # 2. Format historical context summary with explicit relevance rationale
        hist_context_text = "No direct historical analog found in event store."
        if historical_events:
            hist_snippets = []
            for ev in historical_events[:3]:
                ev_id = ev.get("event_id")
                ev_name = ev.get("name")
                scope = ev.get("event_scope")
                relevance = ev.get("relevance", "Topical historical analog")
                obs_trade = ev.get("observed_trade_impact", {}).get("avg_trade_return_1m")
                obs_trade_str = f"{obs_trade:+.2f}%" if obs_trade is not None else "N/A"
                hist_snippets.append(
                    f"- [{ev_id}] {ev_name} (Scope: {scope}) | Relevance: {relevance} | Observed Trade Return: {obs_trade_str}"
                )
            hist_context_text = "\n".join(hist_snippets)

        # 3. Construct LLM prompt for mitigation synthesis
        trade_pred = ml_predictions.get("trade", {}).get("Trade_Return_1M_Pred")
        trade_str = f"{trade_pred:+.2f}%" if trade_pred is not None else "N/A"
        prod_risk = ml_predictions.get("agriculture", {}).get("Production_Risk", "Medium")
        price_pred = ml_predictions.get("price", {}).get("Price_Return_1M_Pred")
        price_text = f"{price_pred:+.2f}%" if price_pred is not None else "Unavailable (Model B Lag1 missing)"
        gva_pred = ml_predictions.get("economy", {}).get("Agri_GVA_Growth_Pred")
        gva_str = f"{gva_pred:+.2f}%" if gva_pred is not None else "N/A"
        infl_pred = ml_predictions.get("economy", {}).get("Inflation_Change_Pred")
        infl_str = f"{infl_pred:+.2f} pp" if infl_pred is not None else "N/A"

        prompt = f"""
Synthesize grounded, realistic agricultural trade resilience and mitigation options for the following scenario:

Scenario Context:
- Target Flow: {trade_flow_desc} for {commodity}
- Event Type: {event_type} | Canonical Shock Direction: {canonical_dir}
- Summary: {event_dict.get('summary', '')}

Econometric ML Forecasts:
- Model A Trade Return: {trade_str}
- Model B Production Risk: {prod_risk}
- Model C Domestic Price Return: {price_text}
- Model D Agricultural GVA Growth: {gva_str} | Inflation Delta: {infl_str}

Historical Precedents from Event Store:
{hist_context_text}

Stakeholder Context:
- Farmers Advisory: {stakeholder_impacts.get('farmers', {}).get('advisory', 'N/A')}
- Consumers Advisory: {stakeholder_impacts.get('consumers', {}).get('advisory', 'N/A')}
- Exporters Advisory: {stakeholder_impacts.get('exporters', {}).get('advisory', 'N/A')}
- Importers Advisory: {stakeholder_impacts.get('importers', {}).get('advisory', 'N/A')}
- Government Advisory: {stakeholder_impacts.get('government', {}).get('advisory', 'N/A')}

Generate a structured JSON object containing 2-3 specific, actionable mitigation options for each stakeholder group:
{{
  "government": [
    "Concrete, realistic policy action 1 (e.g. buffer stock management, tariff calibration)",
    "Concrete, realistic policy action 2"
  ],
  "farmers": [
    "Practical producer mitigation option 1 (e.g. crop diversification, storage financing)",
    "Practical producer mitigation option 2"
  ],
  "consumers": [
    "Practical consumer protection measure 1 (e.g. targeted PDS access, price monitoring)",
    "Practical consumer protection measure 2"
  ],
  "exporters": [
    "Practical export risk mitigation option 1 (e.g. destination diversification, hedging)",
    "Practical export risk mitigation option 2"
  ],
  "importers": [
    "Practical import continuity measure 1 (e.g. alternative origin contracting, inventory buffer)",
    "Practical import continuity measure 2"
  ]
}}

STRICT PUBLICATION RULES:
- Do NOT fabricate fake citations or non-existent specific scheme names.
- If referencing general policy mechanisms (e.g. buffer stocks, Open Market Sales, MSP procurement, tariff rate quotas), present them clearly as exploratory policy options.
"""

        system_instruction = (
            "You are the Drishti Mitigation & Action Agent. Your job is to propose concrete, "
            "policy-grounded mitigation playbooks based on empirical ML forecasts and historical precedents. "
            "Do not fabricate citations or present recommendations as authoritative government mandates."
        )

        mitigation_json = self.llm.generate_structured_json(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        return {
            "historical_context": [
                {
                    "event_id": ev.get("event_id"),
                    "name": ev.get("name"),
                    "event_scope": ev.get("event_scope"),
                    "relevance": ev.get("relevance", "Topical historical analog"),
                    "observed_trade_impact": ev.get("observed_trade_impact"),
                    "observed_price_impact": ev.get("observed_price_impact"),
                    "narrative": ev.get("narrative"),
                    "provenance": "[HISTORICAL EVENT STORE]",
                }
                for ev in historical_events[:3]
            ],
            "mitigation_actions": {
                "government": mitigation_json.get("government", []),
                "farmers": mitigation_json.get("farmers", []),
                "consumers": mitigation_json.get("consumers", []),
                "exporters": mitigation_json.get("exporters", []),
                "importers": mitigation_json.get("importers", []),
                "llm_provider": getattr(self.llm, "last_provider_used", None) or "Offline Synthesis",
                "provenance": "[LLM INFERENCE]",
            },
            "llm_provider": getattr(self.llm, "last_provider_used", None) or "Offline Synthesis",
            "disclaimer": (
                "Mitigation recommendations are exploratory decision support generated by LLM reasoning "
                "over empirical ML cascade forecasts and historical event records. They do not constitute "
                "authoritative government policy mandates."
            ),
        }
