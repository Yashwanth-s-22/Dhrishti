"""
Drishti Agent: Event Intelligence  Agent

=======================================
Extracts, structures, and validates geopolitical and agricultural trade shock events.
Queries the GDELT DOC 2.0 API via MCP for live/recent news context, resolves commodity
to HS4 codes, and structures the shock parameters for downstream quantitative cascade execution.
Strictly separates:
- User / CLI Specified Parameters [USER / CLI PARAMETER]
- Factual Journalistic News [GDELT DATA]
- Classified Event Parameters & Canonical Shock Direction [LLM INFERENCE]
"""

from typing import Dict, Any, Optional
from config.settings import COMMODITY_TO_HS4
from drishti_mcp.drishti_mcp_server import call_mcp_tool
from llm.gemini_client import GeminiClient


class EventIntelligenceAgent:
    """
    Agent responsible for event ingestion, news verification via GDELT MCP tool,
    and structured parameter extraction with canonical shock direction.
    """

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()

    @staticmethod
    def format_trade_flow_description(country: str, trade_type: str) -> str:
        """Ensure precise trade flow terminology (India as baseline reference)."""
        c_upper = country.strip().upper()
        tt_lower = trade_type.strip().lower()
        if tt_lower == "import":
            return f"India's imports from {c_upper}"
        elif tt_lower == "export":
            return f"India's exports to {c_upper}"
        return f"Bilateral trade between India and {c_upper} ({trade_type})"

    def process(
        self,
        query: str,
        partner_country: Optional[str] = None,
        commodity: Optional[str] = None,
        hs4: Optional[int] = None,
        trade_type: Optional[str] = None,
        fetch_news: bool = True,
        timespan: str = "7d",
    ) -> Dict[str, Any]:
        """
        Process user query / event statement into a verified, structured event object.

        Args:
            query: Free-text event description or news query (e.g. "India bans non-basmati rice exports")
            partner_country: Optional pre-specified partner country
            commodity: Optional pre-specified commodity
            hs4: Optional pre-specified HS4 code
            trade_type: Optional 'Export' or 'Import'
            fetch_news: Whether to query GDELT for recent news
            timespan: GDELT search timespan

        Returns:
            Structured event dictionary with clearly separated provenance layers.
        """
        news_result = {}
        articles = []
        if fetch_news and query:
            # Construct a concise search query for GDELT
            search_terms = []
            if partner_country:
                search_terms.append(partner_country)
            if commodity:
                search_terms.append(commodity)
            search_query = " ".join(search_terms) if search_terms else query

            news_result = call_mcp_tool("search_gdelt_news", {
                "query": search_query,
                "timespan": timespan,
                "max_records": 5,
            })
            articles = news_result.get("articles", [])

        # Construct prompt for structured extraction
        articles_context = ""
        if articles:
            articles_context = "\nRecent GDELT News Articles:\n" + "\n".join([
                f"- Title: {a.get('title')} | Source: {a.get('source_country')} | Domain: {a.get('domain')} | URL: {a.get('url')}"
                for a in articles[:5]
            ])
        else:
            articles_context = "\nGDELT News: (No live articles retrieved; relying on scenario description)."

        prompt = f"""
Analyze the following agricultural trade scenario and news context.
Extract the structured parameters into the requested JSON schema.

User Query / Scenario: {query}
{articles_context}

Pre-specified inputs (must preserve if provided):
- Partner Country: {partner_country or 'Infer from context'}
- Commodity: {commodity or 'Infer from context'}
- HS4: {hs4 or 'Infer from commodity'}
- Trade Flow: {trade_type or 'Infer from context (Export or Import from India perspective)'}

Available HS4 reference codes:
{COMMODITY_TO_HS4}

Required JSON schema:
{{
  "country": "PARTNER_COUNTRY_NAME (e.g. RUSSIA, INDONESIA, UNITED STATES, BANGLADESH)",
  "commodity": "COMMODITY_NAME (e.g. Wheat, Rice, Palm Oil, Onion, Soybean)",
  "hs4": 1001,  // 4-digit integer HS4 code
  "trade_type": "Export or Import",
  "event_type": "export_restriction | supply_shock | tariff | conflict | logistics_disruption | policy_shift",
  "shock_direction": "supply_contraction | supply_shock | demand_contraction | demand_shock | trade_restriction | logistics_disruption",
  "approximate_timing": "recent or timeframe string",
  "summary": "2-3 sentence factual synthesis of the trade shock scenario.",
  "confidence": "high | medium | low"
}}
"""

        system_instruction = (
            "You are the Drishti Event Intelligence Agent. Your job is to extract structured, "
            "factual geopolitical and agricultural trade parameters from news and user requests. "
            "STRICT RULES:\n"
            "1. India is the home country; 'trade_type' represents India's trade flow (Export or Import).\n"
            "2. 'country' is always the foreign partner country.\n"
            "3. 'shock_direction' is the canonical shock classification.\n"
            "4. Do not invent factual news. If no news articles were retrieved, extract parameters from the scenario query alone."
        )

        extracted = self.llm.generate_structured_json(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        # Finalize parameters with user pre-specifications taking strict precedence
        country_final = partner_country or extracted.get("country", "RUSSIA")
        commodity_final = commodity or extracted.get("commodity", "Wheat")
        
        # Resolve HS4
        hs4_final = hs4 or extracted.get("hs4")
        if not hs4_final or not isinstance(hs4_final, int):
            c_key = str(commodity_final).lower().strip()
            hs4_final = COMMODITY_TO_HS4.get(c_key, 1001)

        trade_type_final = trade_type or extracted.get("trade_type", "Export")
        if str(trade_type_final).capitalize() not in ("Export", "Import"):
            trade_type_final = "Export"

        canonical_direction = extracted.get("shock_direction") or extracted.get("direction") or "supply_contraction"
        trade_flow_desc = self.format_trade_flow_description(str(country_final), str(trade_type_final))

        summary_val = extracted.get("summary", "")
        if not summary_val or (partner_country and partner_country.upper() not in summary_val.upper()) or (commodity and commodity.lower() not in summary_val.lower()):
            summary_val = f"Trade shock scenario involving {trade_flow_desc} for {commodity_final} (HS4: {hs4_final})."

        return {
            "user_parameters": {
                "country": str(country_final).upper().strip(),
                "commodity": str(commodity_final).strip(),
                "hs4": int(hs4_final),
                "trade_type": str(trade_type_final).capitalize().strip(),
                "trade_flow_description": trade_flow_desc,
                "provenance": "[USER / CLI PARAMETER]",
            },
            "llm_classification": {
                "event_type": extracted.get("event_type", "supply_shock"),
                "shock_direction": canonical_direction,
                "approximate_timing": extracted.get("approximate_timing", "2024"),
                "summary": summary_val,
                "confidence": extracted.get("confidence", "high"),
                "provenance": "[LLM INFERENCE]",
            },
            "event": {
                "country": str(country_final).upper().strip(),
                "commodity": str(commodity_final).strip(),
                "hs4": int(hs4_final),
                "trade_type": str(trade_type_final).capitalize().strip(),
                "trade_flow_description": trade_flow_desc,
                "event_type": extracted.get("event_type", "supply_shock"),
                "shock_direction": canonical_direction,
                "approximate_timing": extracted.get("approximate_timing", "2024"),
                "summary": summary_val,
                "confidence": extracted.get("confidence", "high"),
                "provenance": "[LLM INFERENCE]",
            },
            "event_sources": [
                {
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "source_country": a.get("source_country"),
                    "domain": a.get("domain"),
                    "seen_date": a.get("seen_date"),
                    "provenance": "[GDELT DATA]",
                }
                for a in articles
            ],
            "gdelt_metadata": {
                "status": news_result.get("status", "not_queried"),
                "article_count": len(articles),
                "query": news_result.get("query", ""),
                "error": news_result.get("error"),
                "provenance": "[GDELT DATA]",
            },
        }
