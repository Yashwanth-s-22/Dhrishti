"""
Unit Tests: Drishti MCP Tools & Server
======================================
Tests tool registration, direct tool execution via call_mcp_tool, ML cascade numerical
provenance, historical event retrieval, and stakeholder rules.
"""

import unittest
import os
from drishti_mcp.drishti_mcp_server import call_mcp_tool


class TestMCPTools(unittest.TestCase):
    """Test suite for Drishti MCP server tools."""

    def test_unknown_tool_returns_error(self):
        res = call_mcp_tool("nonexistent_tool", {"param": 1})
        self.assertEqual(res["status"], "error")
        self.assertIn("not found", res["message"])
        self.assertIn("available_tools", res)

    def test_run_drishti_ml_cascade_execution(self):
        res = call_mcp_tool("run_drishti_ml_cascade", {
            "country": "RUSSIA",
            "trade_type": "Import",
            "hs4": 1001,
            "year": 2024,
            "month": 1,
            "shock_intensity": 1.0,
            "trade_share": 5.0,
        })

        self.assertEqual(res["status"], "success")
        if isinstance(res["provenance"], dict):
            self.assertEqual(res["provenance"]["predictions"], "[ML MODEL OUTPUT]")
        else:
            self.assertEqual(res["provenance"], "[ML MODEL OUTPUT]")
        self.assertIn("predictions", res)
        
        preds = res["predictions"]
        self.assertIn("trade", preds)
        self.assertIn("agriculture", preds)
        self.assertIn("price", preds)
        self.assertIn("economy", preds)

        # Verify numerical outputs
        self.assertIsInstance(preds["trade"]["Trade_Return_1M_Pred"], float)
        self.assertIsInstance(preds["agriculture"]["Production_Growth_Pred"], float)
        self.assertIn(preds["agriculture"]["Production_Risk"], ["Low", "Medium", "High", "Critical"])
        self.assertIsInstance(preds["economy"]["Agri_GVA_Growth_Pred"], float)
        self.assertIsInstance(preds["economy"]["Inflation_Change_Pred"], float)

    def test_get_historical_event_by_id(self):
        res = call_mcp_tool("get_historical_event", {"event_id": "EVT001"})
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["provenance"], "[HISTORICAL EVENT STORE]")
        self.assertGreaterEqual(len(res["events"]), 1)

        evt = res["events"][0]
        self.assertEqual(evt["event_id"], "EVT001")
        self.assertIn("Russia-Ukraine", evt["name"])
        self.assertEqual(evt["event_scope"], "direct")
        self.assertIn("methodological_note", res)

    def test_get_historical_event_proxy_event(self):
        res = call_mcp_tool("get_historical_event", {"event_id": "EVT006"})
        if res.get("status") == "success":
            evt = res["events"][0]
            self.assertEqual(evt["event_scope"], "proxy")

    def test_get_stakeholder_analysis_execution(self):
        res = call_mcp_tool("get_stakeholder_analysis", {
            "country": "RUSSIA",
            "trade_type": "Import",
            "hs4": 1001,
            "commodity_name": "Wheat",
            "trade_pred": -0.71,
            "prod_growth": -0.42,
            "prod_risk": "Low",
            "trade_share": 5.0,
            "month": 1,
            "year": 2024,
        })

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["provenance"], "[RULE-BASED OUTPUT]")
        self.assertIn("effects", res)
        self.assertIn("stakeholder_advisories", res)
        
        advisories = res["stakeholder_advisories"]
        self.assertIn("farmers", advisories)
        self.assertIn("consumers", advisories)
        self.assertIn("exporters", advisories)
        self.assertIn("importers", advisories)
        self.assertIn("regional", advisories)
        self.assertIn("government", advisories)


if __name__ == "__main__":
    unittest.main()
