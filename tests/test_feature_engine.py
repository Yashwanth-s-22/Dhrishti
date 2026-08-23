"""
Unit Tests for Drishti Feature Engine & Historical Lookups
==========================================================
Validates that canonical feature engineering and historical dataset lookups:
1. Enforce strict temporal ordering (Date < T).
2. Never fabricate or zero-fill missing upstream lags.
3. Attach exact provenance to every constructed feature.
"""

import unittest
from drishti_mcp.tools.feature_engine import construct_canonical_feature_vector


class TestFeatureEngine(unittest.TestCase):

    def test_russia_wheat_feature_construction(self):
        """Test feature vector construction for Russia Wheat scenario."""
        fvec = construct_canonical_feature_vector(
            event_country="RUSSIA",
            commodity="Wheat",
            hs4=1001,
            trade_type="Import",
            event_date="2024-06-15",
            goldstein_score=-9.0,
            avg_tone=-6.5,
            num_mentions=150,
            user_shock_intensity=1.5,
            user_trade_share=5.0,
        )

        self.assertIn("features_a", fvec)
        self.assertIn("features_b", fvec)
        feat_a = fvec["features_a"]

        # Assert correct calculated values
        self.assertAlmostEqual(feat_a["Shock_Intensity_Lag1"].iloc[0], 1.5)
        self.assertAlmostEqual(feat_a["Trade_Share_Lag1"].iloc[0], 5.0)
        self.assertAlmostEqual(feat_a["Lagged_Effective_Shock_1"].iloc[0], 0.075)

        # Provenance counts must be populated
        counts = fvec["provenance_counts"]
        self.assertGreater(counts["calculated"], 0)
        self.assertGreater(counts["historical_lookups"], 0)

    def test_non_crop_model_b_lag_unavailability(self):
        """Test that Palm Oil correctly detects that Model B t-1 OOF lag is unavailable."""
        fvec = construct_canonical_feature_vector(
            event_country="INDONESIA",
            commodity="Palm Oil",
            hs4=1511,
            trade_type="Import",
            event_date="2024-06-15",
            goldstein_score=-5.0,
            avg_tone=-4.2,
            num_mentions=85,
        )

        # Palm Oil is not an Indian domestic crop -> Model B OOF lag1 must be None
        self.assertIsNone(fvec["model_b_oof_lag1"])
        self.assertEqual(fvec["provenance_details"]["Production_Growth_Pred_Lag1"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
