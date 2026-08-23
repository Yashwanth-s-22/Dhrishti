"""
Unit Tests: Drishti Individual Agents
=====================================
Tests each of the 4 core agent modules in isolation:
1. EventIntelligenceAgent
2. ImpactInterpretationAgent
3. StakeholderAdvisoryAgent
4. MitigationActionAgent
"""

import unittest
from unittest.mock import patch, MagicMock

from llm.gemini_client import GeminiClient
from agents.event_intelligence_agent import EventIntelligenceAgent
from agents.impact_interpretation_agent import ImpactInterpretationAgent
from agents.stakeholder_advisory_agent import StakeholderAdvisoryAgent
from agents.mitigation_action_agent import MitigationActionAgent


class TestAgents(unittest.TestCase):
    """Test suite for Drishti Agent modules."""

    def setUp(self):
        self.mock_llm = GeminiClient(use_mock=True)

    def test_event_intelligence_agent(self):
        agent = EventIntelligenceAgent(self.mock_llm)
        result = agent.process(
            query="Russia bans wheat exports following conflict escalation",
            fetch_news=False,
        )

        self.assertIn("event", result)
        self.assertIn("event_sources", result)
        ev = result["event"]
        self.assertEqual(ev["country"], "RUSSIA")
        self.assertEqual(ev["commodity"], "Wheat")
        self.assertEqual(ev["hs4"], 1001)
        self.assertEqual(ev["provenance"], "[LLM INFERENCE]")

    def test_impact_interpretation_agent(self):
        agent = ImpactInterpretationAgent(self.mock_llm)
        event_dict = {
            "country": "RUSSIA",
            "trade_type": "Import",
            "hs4": 1001,
            "commodity": "Wheat",
        }
        result = agent.process(event_dict, shock_intensity=1.0, trade_share=5.0)

        self.assertIn("ml_predictions", result)
        self.assertIn("economic_interpretation", result)
        
        ml = result["ml_predictions"]
        self.assertEqual(ml["provenance"], "[ML MODEL OUTPUT]")
        self.assertIn("trade", ml)
        self.assertIn("agriculture", ml)
        self.assertIn("price", ml)
        self.assertIn("economy", ml)

    def test_stakeholder_advisory_agent(self):
        agent = StakeholderAdvisoryAgent(self.mock_llm)
        event_dict = {
            "country": "RUSSIA",
            "trade_type": "Import",
            "hs4": 1001,
            "commodity": "Wheat",
        }
        ml_preds = {
            "trade": {"Trade_Return_1M_Pred": -0.71},
            "agriculture": {"Production_Growth_Pred": -0.42, "Production_Risk": "Low"},
            "price": {"Price_Return_1M_Pred": None},
            "economy": {"Agri_GVA_Growth_Pred": 4.64, "Inflation_Change_Pred": -1.72},
        }
        result = agent.process(event_dict, ml_preds, trade_share=5.0)

        self.assertIn("stakeholder_impacts", result)
        self.assertEqual(result["provenance"], "[RULE-BASED OUTPUT]")
        sh = result["stakeholder_impacts"]
        self.assertIn("farmers", sh)
        self.assertIn("consumers", sh)
        self.assertIn("exporters", sh)
        self.assertIn("importers", sh)
        self.assertIn("regional", sh)
        self.assertIn("government", sh)

    def test_mitigation_action_agent(self):
        agent = MitigationActionAgent(self.mock_llm)
        event_dict = {
            "country": "RUSSIA",
            "trade_type": "Import",
            "hs4": 1001,
            "commodity": "Wheat",
            "event_type": "export_restriction",
        }
        ml_preds = {
            "trade": {"Trade_Return_1M_Pred": -0.71},
            "agriculture": {"Production_Growth_Pred": -0.42, "Production_Risk": "Low"},
            "price": {"Price_Return_1M_Pred": None},
            "economy": {"Agri_GVA_Growth_Pred": 4.64, "Inflation_Change_Pred": -1.72},
        }
        stakeholder_impacts = {
            "farmers": {"advisory": "Maintain storage resilience."},
            "consumers": {"advisory": "Monitor food prices."},
            "exporters": {"advisory": "Not applicable for import flow."},
            "importers": {"advisory": "Contract alternative suppliers."},
        }
        result = agent.process(event_dict, ml_preds, stakeholder_impacts)

        self.assertIn("historical_context", result)
        self.assertIn("mitigation_actions", result)
        actions = result["mitigation_actions"]
        self.assertIn("government", actions)
        self.assertIn("farmers", actions)
        self.assertIn("consumers", actions)
        self.assertIn("exporters", actions)
        self.assertIn("importers", actions)
        self.assertEqual(actions["provenance"], "[LLM INFERENCE]")


if __name__ == "__main__":
    unittest.main()
