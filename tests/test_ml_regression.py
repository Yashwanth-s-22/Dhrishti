"""
Regression Test Suite for Drishti Econometric ML Cascade
=========================================================
Ensures that the Agentic and MCP layer modifications have NOT altered or corrupted
the underlying frozen ML model predictions or their temporal OOF lag provenance.
"""

import unittest
import numpy as np
from drishti_mcp.drishti_mcp_server import call_mcp_tool


class TestMLCascadeRegression(unittest.TestCase):
    """Verify numerical invariance of the ML cascade."""

    def test_model_a_numerical_invariance(self):
        """Test Model A trade flow prediction for deterministic baseline input."""
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
        trade_pred = res["predictions"]["trade"]["Trade_Return_1M_Pred"]
        self.assertIsInstance(trade_pred, float)
        # Verify deterministic prediction from historical feature vector
        self.assertAlmostEqual(trade_pred, -0.811358, places=2)

    def test_model_b_numerical_invariance(self):
        """Test Model B agricultural production growth and risk classification."""
        res = call_mcp_tool("run_drishti_ml_cascade", {
            "country": "RUSSIA",
            "trade_type": "Import",
            "hs4": 1001,
            "year": 2024,
            "month": 1,
            "shock_intensity": 1.0,
            "trade_share": 5.0,
        })
        prod_pred = res["predictions"]["agriculture"]["Production_Growth_Pred"]
        prod_risk = res["predictions"]["agriculture"]["Production_Risk"]
        self.assertIsInstance(prod_pred, float)
        self.assertIn(prod_risk, ["Low", "Medium", "High", "Critical"])
        self.assertAlmostEqual(prod_pred, -3.0637, places=2)
        self.assertEqual(prod_risk, "Low")

    def test_missing_model_c_remains_unavailable_not_zero(self):
        """Test that missing Model B lag correctly leaves Model C as UNAVAILABLE for non-crop (Palm Oil HS4 1511)."""
        res = call_mcp_tool("run_drishti_ml_cascade", {
            "country": "INDONESIA",
            "trade_type": "Import",
            "hs4": 1511,
            "year": 2024,
            "month": 1,
            "shock_intensity": 1.0,
            "trade_share": 5.0,
        })
        price_pred = res["predictions"]["price"]["Price_Return_1M_Pred"]
        price_status = res["predictions"]["price"]["status"]
        self.assertIsNone(price_pred)
        self.assertEqual(price_status, "UNAVAILABLE")
        self.assertIn("unavailable", res["predictions"]["price"]["unavailable_reason"].lower())

    def test_model_d_macro_numerical_invariance(self):
        """Test Model D GVA and Inflation predictions."""
        res = call_mcp_tool("run_drishti_ml_cascade", {
            "country": "RUSSIA",
            "trade_type": "Import",
            "hs4": 1001,
            "year": 2024,
            "month": 1,
            "shock_intensity": 1.0,
            "trade_share": 5.0,
        })
        gva_pred = res["predictions"]["economy"]["Agri_GVA_Growth_Pred"]
        infl_pred = res["predictions"]["economy"]["Inflation_Change_Pred"]
        self.assertIsInstance(gva_pred, float)
        self.assertIsInstance(infl_pred, float)
        self.assertAlmostEqual(gva_pred, 4.635821, places=3)
        self.assertAlmostEqual(infl_pred, -1.330406, places=3)

    def test_coverage_and_cascade_state_reporting(self):
        """Test that cascade coverage ratio and partial state are correctly evaluated for non-crop."""
        res = call_mcp_tool("run_drishti_ml_cascade", {
            "country": "INDONESIA",
            "trade_type": "Import",
            "hs4": 1511,
            "year": 2024,
            "month": 1,
        })
        coverage = res["coverage"]
        self.assertEqual(coverage["cascade_state"], "PARTIAL")
        self.assertEqual(coverage["coverage_ratio"], "4/5")
        self.assertEqual(len(coverage["unavailable_stages"]), 1)


if __name__ == "__main__":
    unittest.main()
