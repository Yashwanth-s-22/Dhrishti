"""
Unit Tests for Drishti Scenarios Module
=======================================
Validates that scenarios in scenarios/scenarios.py execute cleanly and produce valid outputs.
"""

import unittest
from scenarios.scenarios import SCENARIOS
from agents.orchestrator import DrishtiAgentOrchestrator


class TestScenarios(unittest.TestCase):

    def setUp(self):
        self.orchestrator = DrishtiAgentOrchestrator()

    def test_scenarios_dictionary_structure(self):
        """Test that SCENARIOS dictionary contains required keys."""
        self.assertIn("wheat_russia_conflict", SCENARIOS)
        self.assertIn("indonesia_palm_oil", SCENARIOS)
        self.assertIn("china_soybean_trade", SCENARIOS)

        for key, sc in SCENARIOS.items():
            self.assertIn("event_country", sc)
            self.assertIn("commodity", sc)
            self.assertIn("hs4", sc)
            self.assertIn("trade_type", sc)
            self.assertIn("event_date", sc)

    def test_scenario_execution_russia_wheat(self):
        """Test full execution of wheat_russia_conflict scenario."""
        res = self.orchestrator.run_scenario_dict("wheat_russia_conflict", SCENARIOS["wheat_russia_conflict"])
        self.assertIn("ml_predictions", res)
        self.assertIn("stakeholder_impacts", res)
        self.assertIn("mitigation_actions", res)

    def test_scenario_execution_indonesia_palm_oil(self):
        """Test full execution of indonesia_palm_oil scenario."""
        res = self.orchestrator.run_scenario_dict("indonesia_palm_oil", SCENARIOS["indonesia_palm_oil"])
        self.assertIn("ml_predictions", res)
        ml = res["ml_predictions"]
        # Model C must be UNAVAILABLE for palm oil
        self.assertEqual(ml["price"]["status"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
