"""
Drishti Agentic Intelligence Orchestrator
=========================================
Central coordinator for the Drishti Decision Intelligence Layer:
Executes the sequential multi-agent workflow:
  Scenario / User Query
      ↓
  Event Intelligence & Canonical Feature Engineering (Historical Lookups)
      ↓
  Impact Interpretation Agent (ML Cascade A -> B -> C -> D)
      ↓
  Stakeholder Advisory Agent (Deterministic Rule Engine)
      ↓
  Mitigation & Action Agent (Historical Event Precedents & Decision Playbooks)
      ↓
  Clean Decision Intelligence Report
"""

import os
import sys
import json
import argparse
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from llm.gemini_client import GeminiClient
from agents.event_intelligence_agent import EventIntelligenceAgent
from agents.impact_interpretation_agent import ImpactInterpretationAgent
from agents.stakeholder_advisory_agent import StakeholderAdvisoryAgent
from agents.mitigation_action_agent import MitigationActionAgent
from drishti_mcp.tools.ml_cascade_tool import run_drishti_ml_cascade


def _wrap_text(text: str, width: int = 84, indent: str = "  ") -> str:
    """Helper to wrap long paragraphs cleanly for terminal display with Windows console safe encoding."""
    if not text:
        return ""
    # Sanitize common non-ascii unicode quotes and hyphens for Windows cp1252 console
    sanitized = (
        text.replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
    )
    paras = sanitized.split("\n")
    wrapped_paras = []
    for p in paras:
        if p.strip():
            wrapped = textwrap.fill(p.strip(), width=width, initial_indent=indent, subsequent_indent=indent)
            wrapped_paras.append(wrapped)
        else:
            wrapped_paras.append("")
    return "\n".join(wrapped_paras)


class DrishtiAgentOrchestrator:
    """
    Central pipeline orchestrator coordinating the Drishti Agentic Layer.
    """

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()
        self.event_agent = EventIntelligenceAgent(self.llm)
        self.impact_agent = ImpactInterpretationAgent(self.llm)
        self.stakeholder_agent = StakeholderAdvisoryAgent(self.llm)
        self.mitigation_agent = MitigationActionAgent(self.llm)

    def run_scenario_dict(self, scenario_name: str, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full inference pipeline directly from a predefined scenario dictionary.
        """
        event_country = scenario_data.get("event_country", "RUSSIA")
        commodity = scenario_data.get("commodity", "Wheat")
        hs4 = int(scenario_data.get("hs4", 1001))
        trade_type = scenario_data.get("trade_type", "Import")
        event_date = scenario_data.get("event_date", "2024-06-15")
        goldstein = scenario_data.get("goldstein_score", -5.0)
        avg_tone = scenario_data.get("avg_tone", -4.0)
        mentions = scenario_data.get("num_mentions", 50)
        event_root = scenario_data.get("event_root", "16")
        event_code = scenario_data.get("event_code", "163")
        shock_intensity = scenario_data.get("shock_intensity")
        trade_share = scenario_data.get("trade_share")

        # Step 1: Execute quantitative ML cascade with canonical feature engineering
        print("  [*] [1/3] Executing canonical feature engineering and quantitative ML cascade...", flush=True)
        cascade_output = run_drishti_ml_cascade(
            country=event_country,
            trade_type=trade_type,
            hs4=hs4,
            commodity=commodity,
            event_date=event_date,
            shock_intensity=shock_intensity,
            trade_share=trade_share,
            goldstein_score=goldstein,
            avg_tone=avg_tone,
            num_mentions=mentions,
            event_root=event_root,
            event_code=event_code,
        )

        ml_predictions = cascade_output["predictions"]
        prov_counts = cascade_output.get("provenance_counts", {})
        prov_details = cascade_output.get("provenance_details", {})

        # Step 2: Economic Interpretation
        print("  [*] [2/3] Generating non-causal econometric interpretation...", flush=True)
        flow_direction_desc = f"India's exports to {event_country}" if trade_type.lower() == "export" else f"India's imports from {event_country}"
        interp_prompt = (
            f"Synthesize the economic implications of the following quantitative forecasts for {flow_direction_desc} "
            f"of {commodity} (HS4: {hs4}) [Trade Flow Direction: {trade_type}]:\n"
            f"- Model A Trade Return 1M: {ml_predictions['trade']['Trade_Return_1M_Pred']:+.2f}%\n"
            f"- Model B National Production Growth: {ml_predictions['agriculture']['Production_Growth_Pred']:+.2f}%, Risk: {ml_predictions['agriculture']['Production_Risk']}\n"
            f"- Model C Price Return 1M: {ml_predictions['price']['Price_Return_1M_Pred'] if ml_predictions['price']['status'] == 'AVAILABLE' else 'UNAVAILABLE'}\n"
            f"- Model D Agri GVA Growth: {ml_predictions['economy']['Agri_GVA_Growth_Pred']:+.2f}%, Food Inflation Delta: {ml_predictions['economy']['Inflation_Change_Pred']:+.2f} pp\n\n"
            f"TRADE DIRECTION CONSISTENCY RULES:\n"
            f"- Trade Flow is '{trade_type}'.\n"
            f"- If Trade Flow is Export: Model A negative trade return represents weaker export flows / export contraction to {event_country}; positive return represents export expansion.\n"
            f"- If Trade Flow is Import: Model A negative trade return represents weaker import flows / import contraction from {event_country}; positive return represents import surge.\n"
            f"- Strictly respect the '{trade_type}' direction from India's baseline perspective."
        )
        economic_interp_text = self.llm.generate_text(
            prompt=interp_prompt,
            system_instruction="You are an econometric analyst. Never make causal claims. Use non-causal econometric phrasing (correlates, associates, forecasts). Strictly respect trade flow direction (Export vs Import)."
        )
        interp_provider = getattr(self.llm, "last_provider_used", None) or "Groq — openai/gpt-oss-120b"

        # Step 3: Stakeholder Engine & Mitigation Playbooks
        print("  [*] [3/3] Evaluating stakeholder engine and synthesizing grounded mitigation playbooks...", flush=True)
        structured_event = {
            "country": event_country,
            "commodity": commodity,
            "hs4": hs4,
            "trade_type": trade_type,
            "event_type": "geopolitical_shock",
            "shock_direction": "supply_contraction",
            "approximate_timing": event_date,
            "summary": scenario_data.get("description", "Geopolitical agricultural trade shock scenario"),
        }

        stakeholder_output = self.stakeholder_agent.process(
            event_dict=structured_event,
            ml_predictions=ml_predictions,
            trade_share=trade_share or 5.0,
            effective_shock=float((shock_intensity or 1.0) * ((trade_share or 5.0) / 100.0)),
            month=int(event_date.split("-")[1]) if "-" in event_date else 6,
            year=int(event_date.split("-")[0]) if "-" in event_date else 2024,
        )

        mitigation_output = self.mitigation_agent.process(
            event_dict=structured_event,
            ml_predictions=ml_predictions,
            stakeholder_impacts=stakeholder_output["stakeholder_impacts"],
        )
        mitigation_provider = mitigation_output.get("llm_provider") or getattr(self.mitigation_agent.llm, "last_provider_used", None) or "Groq — openai/gpt-oss-120b"

        final_result = {
            "scenario_name": scenario_name,
            "scenario_data": scenario_data,
            "ml_predictions": ml_predictions,
            "provenance_counts": prov_counts,
            "provenance_details": prov_details,
            "economic_interpretation": economic_interp_text,
            "stakeholder_impacts": stakeholder_output["stakeholder_impacts"],
            "historical_context": mitigation_output["historical_context"],
            "mitigation_actions": mitigation_output["mitigation_actions"],
            "llm_usage": {
                "economic_interpretation": interp_provider,
                "mitigation": mitigation_provider,
            },
        }

        print_scenario_report(final_result)
        return final_result

    def run(
        self,
        query: str,
        partner_country: Optional[str] = None,
        commodity: Optional[str] = None,
        hs4: Optional[int] = None,
        trade_type: Optional[str] = None,
        shock_intensity: float = 1.0,
        trade_share: float = 5.0,
        year: int = 2024,
        month: int = 1,
        fetch_news: bool = False,
        timespan: str = "7d",
    ) -> Dict[str, Any]:
        """
        Execute pipeline for interactive CLI arguments.
        """
        event_output = self.event_agent.process(
            query=query,
            partner_country=partner_country,
            commodity=commodity,
            hs4=hs4,
            trade_type=trade_type,
            fetch_news=fetch_news,
            timespan=timespan,
        )
        user_params = event_output["user_parameters"]
        llm_class = event_output["llm_classification"]
        structured_event = event_output["event"]

        impact_output = self.impact_agent.process(
            event_dict=structured_event,
            shock_intensity=shock_intensity,
            trade_share=trade_share,
            year=year,
            month=month,
        )
        ml_predictions = impact_output["ml_predictions"]
        economic_interp = impact_output["economic_interpretation"]

        effective_shock = float(shock_intensity * (trade_share / 100.0))

        stakeholder_output = self.stakeholder_agent.process(
            event_dict=structured_event,
            ml_predictions=ml_predictions,
            trade_share=trade_share,
            effective_shock=effective_shock,
            month=month,
            year=year,
        )
        stakeholder_impacts = stakeholder_output["stakeholder_impacts"]

        mitigation_output = self.mitigation_agent.process(
            event_dict=structured_event,
            ml_predictions=ml_predictions,
            stakeholder_impacts=stakeholder_impacts,
        )

        return {
            "query": query,
            "user_parameters": user_params,
            "llm_classification": llm_class,
            "event": structured_event,
            "gdelt_metadata": event_output["gdelt_metadata"],
            "event_sources": event_output["event_sources"],
            "ml_predictions": ml_predictions,
            "observed_metrics": impact_output["observed_metrics"],
            "economic_interpretation": economic_interp,
            "stakeholder_impacts": stakeholder_impacts,
            "historical_context": mitigation_output["historical_context"],
            "mitigation_actions": mitigation_output["mitigation_actions"],
            "provenance": {
                "event_sources": "[GDELT DATA]",
                "user_parameters": "[USER / CLI PARAMETER]",
                "llm_classification": "[LLM INFERENCE]",
                "ml_predictions": "[ML MODEL OUTPUT]",
                "observed_metrics": "[OBSERVED DATASET RECORD]",
                "economic_interpretation": "[LLM INFERENCE]",
                "stakeholder_impacts": "[RULE-BASED OUTPUT]",
                "historical_context": "[HISTORICAL EVENT STORE]",
                "mitigation_actions": "[LLM INFERENCE]",
            },
            "disclaimer": (
                "Drishti econometric predictions represent statistical associations and walk-forward "
                "scenario forecasts, not proven causal mechanisms. LLMs provide non-causal synthesis "
                "and exploratory decision playbooks."
            ),
        }


# Alias for compatibility
DrishtiOrchestrator = DrishtiAgentOrchestrator


def print_scenario_report(result: Dict[str, Any]):
    """
    Format clean terminal report per Drishti Section 14 Specification.
    """
    sc = result.get("scenario_data", {})
    ml = result.get("ml_predictions", {})
    trade = ml.get("trade", {})
    agri = ml.get("agriculture", {})
    price = ml.get("price", {})
    econ = ml.get("economy", {})
    cov = ml.get("coverage", {})
    counts = result.get("provenance_counts", {})

    div_heavy = "=" * 74
    div_light = "-" * 74

    print("\n" + div_heavy)
    print("                      DRISHTI DECISION INTELLIGENCE                       ")
    print(div_heavy)

    print("\nEVENT")
    print(div_light)
    print(f"  Date          : {sc.get('event_date', 'N/A')}")
    print(f"  Country       : {sc.get('event_country', 'N/A')}")
    print(f"  Event Code    : {sc.get('event_code', 'N/A')} (Root: {sc.get('event_root', 'N/A')})")
    print(f"  Goldstein     : {sc.get('goldstein_score', 'N/A')}")
    print(f"  Avg Tone      : {sc.get('avg_tone', 'N/A')}")

    print("\nSCENARIO CONTEXT")
    print(div_light)
    print(f"  Commodity     : {sc.get('commodity', 'N/A')}")
    print(f"  HS4           : {sc.get('hs4', 'N/A')}")
    print(f"  Trade Type    : {sc.get('trade_type', 'N/A')}")
    print(f"  Trade Share   : {sc.get('trade_share', 'N/A')}%")
    print(f"  Shock         : {sc.get('shock_intensity', 'N/A')}")

    print("\nQUANTITATIVE ML CASCADE")
    print(div_light)
    t_val = trade.get('Trade_Return_1M_Pred')
    t_str = f"{t_val:+.2f}%" if t_val is not None else "--"
    print(f"  Model A — Trade Return 1M       : {t_str}")

    b_val = agri.get('Production_Growth_Pred')
    b_str = f"{b_val:+.2f}%" if b_val is not None else "--"
    print(f"  Model B — Production Growth     : {b_str}")

    r_val = str(agri.get('Production_Risk', 'LOW')).upper()
    print(f"  Model B — Production Risk       : {r_val}")

    p_val = price.get('Price_Return_1M_Pred')
    p_str = f"{p_val:+.2f}%" if p_val is not None else "--"
    print(f"  Model C — Price Return 1M       : {p_str}")

    g_val = econ.get('Agri_GVA_Growth_Pred')
    g_str = f"{g_val:+.2f}%" if g_val is not None else "--"
    print(f"  Model D — Agri GVA Growth       : {g_str}")

    i_val = econ.get('Inflation_Change_Pred')
    i_str = f"{i_val:+.2f} pp" if i_val is not None else "--"
    print(f"  Model D — Food Inflation 3M     : {i_str}")

    c_state = cov.get("cascade_state", "PARTIAL")
    c_ratio = cov.get("coverage_ratio", "4/5")
    print(f"\n  Cascade Status: A -> B -> C -> D [{c_state} ({c_ratio})]")

    print("\n" + div_heavy)
    print("                    STAKEHOLDER IMPACT                    ")
    print(div_heavy)

    tt = sc.get("trade_type", "Import").capitalize()
    comm = sc.get("commodity", "Commodity")
    dt_str = sc.get("event_date", "2024-06-15")
    m_int = int(dt_str.split("-")[1]) if "-" in dt_str else 6
    season_name = "Kharif" if m_int in [6, 7, 8, 9, 10] else ("Rabi" if m_int in [11, 12, 1, 2, 3] else "Summer")

    # 1. FARMERS
    print("\nFARMERS")
    print(div_light)
    if p_val is not None:
        if p_val < 0:
            f_imp = "NEGATIVE"
            f_why = f"Lower {comm.lower()} prices may reduce producer price realization."
        else:
            f_imp = "POSITIVE"
            f_why = f"Higher {comm.lower()} prices improve producer price realization."
        f_sev = "LOW" if abs(p_val) < 1.0 else ("MODERATE" if abs(p_val) < 5.0 else "HIGH")
        f_conf = "MEDIUM"
        print(f"Impact       : {f_imp}")
        print(f"Severity     : {f_sev}")
        print(f"Confidence   : {f_conf}")
        print("\nWhy?")
        print(_wrap_text(f_why, width=74, indent=""))
        print("\nKey Evidence")
        print(f"• Model C Price Return 1M : {p_str}")
        print(f"• Model B Production Risk : {r_val}")
        print(f"• Production Growth       : {b_str}")
    else:
        f_imp = "NEUTRAL"
        f_sev = "LOW"
        f_conf = "MEDIUM"
        f_why = f"Domestic agricultural model projects stable production outlook for {comm.lower()}."
        print(f"Impact       : {f_imp}")
        print(f"Severity     : {f_sev}")
        print(f"Confidence   : {f_conf}")
        print("\nWhy?")
        print(_wrap_text(f_why, width=74, indent=""))
        print("\nKey Evidence")
        print(f"• Model B Production Risk : {r_val}")
        print(f"• Production Growth       : {b_str}")

    # 2. CONSUMERS
    print("\n\nCONSUMERS")
    print(div_light)
    if p_val is not None:
        if p_val <= 0:
            c_imp = "POSITIVE"
            c_why = f"The projected near-term decline in {comm.lower()} prices could reduce consumer food expenditure."
        else:
            c_imp = "NEGATIVE"
            c_why = f"Projected upward price pressure on {comm.lower()} could increase consumer food expenditure."
        c_sev = "LOW" if abs(p_val) < 1.0 else ("MODERATE" if abs(p_val) < 5.0 else "HIGH")
        c_conf = "MEDIUM"
        print(f"Impact       : {c_imp}")
        print(f"Severity     : {c_sev}")
        print(f"Confidence   : {c_conf}")
        print("\nWhy?")
        print(_wrap_text(c_why, width=74, indent=""))
        print("\nKey Evidence")
        print(f"• Model C Price Return 1M : {p_str}")
        print(f"• Food Inflation 3M Delta : {i_str}")
    else:
        c_imp = "POSITIVE" if (i_val is not None and i_val <= 0) else "NEUTRAL"
        c_sev = "LOW"
        c_conf = "MEDIUM"
        c_why = f"Macroeconomic forecasts indicate stable headline food inflation trends."
        print(f"Impact       : {c_imp}")
        print(f"Severity     : {c_sev}")
        print(f"Confidence   : {c_conf}")
        print("\nWhy?")
        print(_wrap_text(c_why, width=74, indent=""))
        print("\nKey Evidence")
        print(f"• Food Inflation 3M Delta : {i_str}")

    # 3. EXPORTERS
    print("\n\nEXPORTERS")
    print(div_light)
    if tt == "Import":
        print("Impact       : NOT APPLICABLE\n")
        print("Reason")
        print("The scenario represents an import flow into India.")
    else:
        e_imp = "NEGATIVE" if (t_val is not None and t_val < 0) else "POSITIVE"
        e_sev = "MODERATE" if (t_val is not None and abs(t_val) >= 0.5) else "LOW"
        print(f"Impact       : {e_imp}")
        print(f"Severity     : {e_sev}")
        print(f"Confidence   : HIGH")
        print("\nWhy?")
        exp_why = (
            f"The projected trade return ({t_str}) signals export shipment contraction for Indian {comm.lower()} to {sc.get('event_country', 'partner')}."
            if (t_val is not None and t_val < 0)
            else f"The projected trade return ({t_str}) indicates positive export momentum for Indian {comm.lower()}."
        )
        print(_wrap_text(exp_why, width=74, indent=""))
        print("\nKey Evidence")
        print(f"• Model A Trade Return 1M : {t_str}")
        print(f"• Trade Flow              : Export")

    # 4. IMPORTERS
    print("\n\nIMPORTERS")
    print(div_light)
    if tt == "Import":
        imp_imp = "NEGATIVE" if (t_val is not None and t_val < 0) else "POSITIVE"
        imp_sev = "MODERATE" if (t_val is not None and abs(t_val) >= 0.5) else "LOW"
        print(f"Impact       : {imp_imp}")
        print(f"Severity     : {imp_sev}")
        print(f"Confidence   : HIGH")
        print("\nWhy?")
        imp_why = (
            f"The projected import contraction indicates potential supply-chain tightness for Indian {comm.lower()} importers from {sc.get('event_country', 'partner')}."
            if (t_val is not None and t_val < 0)
            else f"The projected import flow indicates steady procurement arrivals for Indian {comm.lower()}."
        )
        print(_wrap_text(imp_why, width=74, indent=""))
        print("\nKey Evidence")
        print(f"• Model A Trade Return 1M : {t_str}")
        print(f"• Trade Flow              : Import")
    else:
        print("Impact       : NOT APPLICABLE\n")
        print("Reason")
        print("The scenario represents an export flow from India.")

    # 5. REGIONAL
    print("\n\nREGIONAL")
    print(div_light)
    print(f"Risk         : {r_val}")
    print(f"Confidence   : HIGH")
    print("\nWhy?")
    print(_wrap_text(f"The agricultural model projects {'positive' if (b_val is not None and b_val >= 0) else 'negative'} production growth with a {r_val.title()} Production Risk classification.", width=74, indent=""))
    print("\nKey Evidence")
    print(f"• Production Growth       : {b_str}")
    print(f"• Production Risk         : {r_val}")
    print(f"• Season                  : {season_name}")

    # 6. GOVERNMENT
    print("\n\nGOVERNMENT")
    print(div_light)
    print("Overall Risk : LOW-MODERATE")
    print("Confidence   : HIGH")
    print("\nKey Indicators")
    print(f"• Trade Return 1M        : {t_str}")
    print(f"• Production Growth      : {b_str}")
    if p_val is not None:
        print(f"• Price Return 1M        : {p_str}")
    print(f"• Agricultural GVA       : {g_str}")
    print(f"• Food Inflation Delta   : {i_str}")
    print("\nInterpretation")
    if tt == "Export":
        flow_text = "weaker export flows" if (t_val is not None and t_val < 0) else ("expanding export shipments" if (t_val is not None and t_val > 0) else "stable export trade flows")
    else:
        flow_text = "weaker import flows" if (t_val is not None and t_val < 0) else ("higher import receipts" if (t_val is not None and t_val > 0) else "stable import trade flows")

    prod_text = "domestic production remains resilient" if (b_val is not None and b_val >= 0) else "domestic production faces contraction pressure"

    if i_val is not None:
        if i_val > 0.05:
            infl_text = f"projected food inflation increases only moderately ({i_str})"
        elif i_val < -0.05:
            infl_text = f"projected food inflation moderates ({i_str})"
        else:
            infl_text = "projected food inflation remains largely stable"
    else:
        infl_text = "macroeconomic inflationary pressure remains contained"

    gov_interp = f"The scenario indicates {flow_text}, while {prod_text} and {infl_text}."
    print(_wrap_text(gov_interp, width=74, indent=""))

    print("\n" + div_heavy)
    print("MITIGATION & ACTIONS")
    print(div_heavy)
    mit = result.get("mitigation_actions", {})
    for role in ["government", "farmers", "consumers", "exporters", "importers"]:
        actions_list = mit.get(role, [])
        if actions_list:
            print(f"\n{role.capitalize()}:")
            for a in actions_list:
                print(_wrap_text(f"- {a}", width=80, indent="  "))

    # 6. LLM USAGE
    llm_use = result.get("llm_usage", {})
    econ_llm = llm_use.get("economic_interpretation", "Groq — openai/gpt-oss-120b")
    mit_llm = llm_use.get("mitigation", "Groq — openai/gpt-oss-120b")

    print("\n" + div_light)
    print("LLM USAGE")
    print(div_light)
    print(f"  Economic Interpretation : {econ_llm}")
    print(f"  Mitigation              : {mit_llm}")
    print(div_heavy + "\n")


def print_formatted_report(result: Dict[str, Any]):
    """Standard formatted report for interactive CLI."""
    div_heavy = "+==========================================================================================+"
    div_light = "------------------------------------------------------------------------------------------"

    print("\n" + div_heavy)
    print("|                          DRISHTI DECISION INTELLIGENCE REPORT                            |")
    print("|       Macro-Agricultural Trade Shock Forecasting & Stakeholder Resilience Playbook       |")
    print(div_heavy + "\n")

    u_params = result.get("user_parameters", {})
    l_class = result.get("llm_classification", {})
    print("01 EVENT INTELLIGENCE")
    print(div_light)
    print(f"  Target Commodity:     {u_params.get('commodity', 'N/A')} (HS4: {u_params.get('hs4', 'N/A')})")
    print(f"  Partner Country:      {u_params.get('country', 'N/A')}")
    print(f"  Trade Flow Context:   {u_params.get('trade_flow_description', 'N/A')}")
    print(f"  Event Classification: {l_class.get('event_type', 'N/A')} | Direction: {l_class.get('shock_direction', 'N/A')}")
    print(f"  Confidence Tier:      {str(l_class.get('confidence', 'HIGH')).upper()}")
    print("\n  Scenario Summary:")
    print(_wrap_text(l_class.get('summary', 'N/A'), width=84, indent="    "))
    print()

    ml = result.get("ml_predictions", {})
    trade = ml.get("trade", {})
    agri = ml.get("agriculture", {})
    price = ml.get("price", {})
    econ = ml.get("economy", {})
    cov = ml.get("coverage", {})

    print("02 QUANTITATIVE ML CASCADE FORECASTS")
    print(div_light)
    print(f"  {'MODEL TARGET':<36} | {'PREDICTION':<14} | {'UNIT':<16} | {'STATUS'}")
    print("  " + "-" * 84)

    t_val = trade.get('Trade_Return_1M_Pred')
    t_str = f"{t_val:+.2f}%" if t_val is not None else "--"
    print(f"  {'Model A: Trade Flow Return 1M':<36} | {t_str:<14} | {'percent':<16} | [AVAILABLE]")

    b_val = agri.get('Production_Growth_Pred')
    b_str = f"{b_val:+.2f}%" if b_val is not None else "--"
    print(f"  {'Model B: National Production Growth':<36} | {b_str:<14} | {'percent':<16} | [AVAILABLE]")

    r_val = str(agri.get('Production_Risk', 'Medium')).upper()
    print(f"  {'Model B: National Production Risk':<36} | {r_val:<14} | {'risk category':<16} | [AVAILABLE]")

    p_val = price.get('Price_Return_1M_Pred')
    p_str = f"{p_val:+.2f}%" if p_val is not None else "--"
    p_stat = "[AVAILABLE]" if p_val is not None else "[--]"
    print(f"  {'Model C: Domestic Price Return 1M':<36} | {p_str:<14} | {'percent':<16} | {p_stat}")

    g_val = econ.get('Agri_GVA_Growth_Pred')
    g_str = f"{g_val:+.2f}%" if g_val is not None else "--"
    print(f"  {'Model D: Agricultural GVA Growth':<36} | {g_str:<14} | {'percent':<16} | [AVAILABLE]")

    i_val = econ.get('Inflation_Change_Pred')
    i_str = f"{i_val:+.2f} pp" if i_val is not None else "--"
    print(f"  {'Model D: Food Inflation 3M Delta':<36} | {i_str:<14} | {'percentage pts':<16} | [AVAILABLE]")
    print("  " + "-" * 84)

    c_state = cov.get("cascade_state", "PARTIAL")
    c_ratio = cov.get("coverage_ratio", "4/5")
    print(f"  Cascade Coverage:     {c_ratio} outputs available | Cascade State: {c_state}\n")

    interp = result.get("economic_interpretation", {})
    print("03 ECONOMIC IMPACT INTERPRETATION (NON-CAUSAL ECONOMETRIC SYNTHESIS)")
    print(div_light)
    print(_wrap_text(interp.get("summary", "N/A"), width=86, indent="  "))
    print()

    sh = result.get("stakeholder_impacts", {})
    print("04 STAKEHOLDER DISAGGREGATION ADVISORIES (DETERMINISTIC RULE ENGINE)")
    print(div_light)
    for role, item in sh.items():
        adv = item.get("advisory", "N/A")
        print(f"  -> {role.upper()}:")
        print(_wrap_text(adv, width=84, indent="     "))
        print()

    mit = result.get("mitigation_actions", {})
    print("05 MITIGATION & ACTION PLAYBOOKS (EXPLORATORY DECISION SUPPORT)")
    print(div_light)
    for role in ["government", "farmers", "consumers", "exporters", "importers"]:
        actions_list = mit.get(role, [])
        if actions_list:
            print(f"  -> For {role.capitalize()}:")
            for a in actions_list:
                print(f"     - {a}")
            print()
    print(div_heavy + "\n")


def main():
    parser = argparse.ArgumentParser(description="Drishti Agentic Intelligence Orchestrator")
    parser.add_argument("--scenario", "-s", type=str, default=None, help="Predefined scenario name from scenarios/scenarios.py")
    parser.add_argument("--query", type=str, default="Russia wheat export disruption and Black Sea grain corridor tension", help="Event query or headline")
    parser.add_argument("--country", type=str, default=None, help="Partner country (e.g. RUSSIA, INDONESIA)")
    parser.add_argument("--commodity", type=str, default=None, help="Commodity name (e.g. Wheat, Rice, Palm Oil)")
    parser.add_argument("--hs4", type=int, default=None, help="4-digit HS4 code (e.g. 1001, 1006)")
    parser.add_argument("--trade-type", type=str, default=None, help="Trade type ('Export' or 'Import')")
    parser.add_argument("--shock-intensity", type=float, default=1.0, help="Shock intensity (default: 1.0)")
    parser.add_argument("--trade-share", type=float, default=5.0, help="Partner trade share % (default: 5.0)")
    parser.add_argument("--year", type=int, default=2024, help="Year (default: 2024)")
    parser.add_argument("--month", type=int, default=1, help="Month (default: 1)")

    args = parser.parse_args()

    orchestrator = DrishtiAgentOrchestrator()

    if args.scenario:
        from scenarios.scenarios import SCENARIOS
        if args.scenario in SCENARIOS:
            orchestrator.run_scenario_dict(args.scenario, SCENARIOS[args.scenario])
            return

    result = orchestrator.run(
        query=args.query,
        partner_country=args.country,
        commodity=args.commodity,
        hs4=args.hs4,
        trade_type=args.trade_type,
        shock_intensity=args.shock_intensity,
        trade_share=args.trade_share,
        year=args.year,
        month=args.month,
        fetch_news=False,
    )
    print_formatted_report(result)


if __name__ == "__main__":
    main()
