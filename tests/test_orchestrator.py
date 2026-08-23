"""
Integration Tests: Drishti Agentic Orchestrator
===============================================
Tests end-to-end multi-agent orchestration, machine-readable JSON structure,
provenance completeness, and missing-data handling.
"""

import unittest
from llm.gemini_client import GeminiClient
from agents.orchestrator import DrishtiAgentOrchestrator


class TestOrchestrator(unittest.TestCase):
    """Test suite for full agentic decision intelligence orchestrator."""

    def setUp(self):
        self.mock_llm = GeminiClient(use_mock=True)
        self.orchestrator = DrishtiAgentOrchestrator(self.mock_llm)

    def test_full_orchestration_workflow(self):
        result = self.orchestrator.run(
            query="Russia wheat export disruption",
            partner_country="RUSSIA",
            commodity="Wheat",
            hs4=1001,
            trade_type="Import",
            shock_intensity=1.0,
            trade_share=5.0,
            year=2024,
            month=1,
            fetch_news=False,
        )

        # 1. Structure Verification
        expected_top_keys = [
            "query",
            "event",
            "event_sources",
            "ml_predictions",
            "economic_interpretation",
            "stakeholder_impacts",
            "historical_context",
            "mitigation_actions",
            "provenance",
            "disclaimer",
        ]
        for key in expected_top_keys:
            self.assertIn(key, result, f"Missing key in final intelligence output: {key}")

        # 2. Event Layer
        ev = result["event"]
        self.assertEqual(ev["country"], "RUSSIA")
        self.assertEqual(ev["commodity"], "Wheat")
        self.assertEqual(ev["hs4"], 1001)
        self.assertEqual(ev["trade_type"], "Import")
        self.assertEqual(ev["provenance"], "[LLM INFERENCE]")

        # 3. ML Layer
        ml = result["ml_predictions"]
        self.assertEqual(ml["provenance"], "[ML MODEL OUTPUT]")
        self.assertIn("Trade_Return_1M_Pred", ml["trade"])
        self.assertIn("Production_Growth_Pred", ml["agriculture"])
        self.assertIn("Production_Risk", ml["agriculture"])
        self.assertIn("Agri_GVA_Growth_Pred", ml["economy"])
        self.assertIn("Inflation_Change_Pred", ml["economy"])

        # 4. Stakeholder Layer
        sh = result["stakeholder_impacts"]
        for role in ["farmers", "consumers", "exporters", "importers", "regional", "government"]:
            self.assertIn(role, sh)
            self.assertEqual(sh[role]["provenance"], "[RULE-BASED OUTPUT]")

        # 5. Mitigation Layer
        actions = result["mitigation_actions"]
        self.assertEqual(actions["provenance"], "[LLM INFERENCE]")
        for role in ["government", "farmers", "consumers", "exporters", "importers"]:
            self.assertIn(role, actions)

        # 6. Provenance Mapping
        prov = result["provenance"]
        self.assertEqual(prov["event_sources"], "[GDELT DATA]")
        self.assertEqual(prov["user_parameters"], "[USER / CLI PARAMETER]")
        self.assertEqual(prov["llm_classification"], "[LLM INFERENCE]")
        self.assertEqual(prov["ml_predictions"], "[ML MODEL OUTPUT]")
        self.assertEqual(prov["observed_metrics"], "[OBSERVED DATASET RECORD]")
        self.assertEqual(prov["economic_interpretation"], "[LLM INFERENCE]")
        self.assertEqual(prov["stakeholder_impacts"], "[RULE-BASED OUTPUT]")
        self.assertEqual(prov["historical_context"], "[HISTORICAL EVENT STORE]")
        self.assertEqual(prov["mitigation_actions"], "[LLM INFERENCE]")

    def test_partial_cascade_missing_lag_handling(self):
        # Evaluate on a commodity with no crop production data (e.g. HS4 0101 Horses/Live Animals)
        result = self.orchestrator.run(
            query="Global live animal trade tariffs",
            partner_country="UNITED STATES",
            commodity="Live Animals",
            hs4=101,
            trade_type="Import",
            fetch_news=False,
        )

        ml = result["ml_predictions"]
        self.assertIn("PARTIAL", ml.get("cascade_state", ""))
        self.assertIsNone(ml["price"]["Price_Return_1M_Pred"])
        self.assertIn("UNAVAILABLE", ml["price"]["status"])


if __name__ == "__main__":
    unittest.main()
