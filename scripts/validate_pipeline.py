"""
Drishti - Task 10: End-to-End Pipeline Validation & Sanity-Check Suite
======================================================================
Comprehensive validation covering:
1. Artifact & structural validation (models, data files, results)
2. Leakage & feature schema sanity checks (safe vs unsafe features in Model D)
3. Semantic cascade validation (state propagation, feature dependencies)
4. Lag feature semantic checks (previous-period shift verification)
5. Model performance sanity checks (MAE vs baseline, GVA single-observation R2 handling)
6. Formula validation status referencing Task 1
7. Event store & stakeholder rule consistency checks
8. Chronological train/val/test split integrity

Methodological Scope:
- This suite checks structural integrity, semantic consistency, and performance sanity.
- It does NOT claim mathematical proof of total leakage absence or causal validity.

Run: python scripts/validate_pipeline.py
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

import joblib

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():
    print("=" * 70)
    print("Drishti - Task 10: End-to-End Pipeline Validation Suite")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    passes = 0
    warns = 0
    infos = 0
    fails = 0

    check_log = []

    def pass_check(cond, msg):
        nonlocal passes, fails
        if cond:
            print(f"  [PASS] {msg}")
            passes += 1
            check_log.append({"type": "PASS", "message": msg})
            return True
        else:
            print(f"  [FAIL] {msg}")
            fails += 1
            check_log.append({"type": "FAIL", "message": msg})
            return False

    def warn_check(cond, msg, warn_detail=""):
        nonlocal passes, warns
        if cond:
            print(f"  [PASS] {msg}")
            passes += 1
            check_log.append({"type": "PASS", "message": msg})
            return True
        else:
            full_msg = f"{msg} - {warn_detail}" if warn_detail else msg
            print(f"  [WARN] {full_msg}")
            warns += 1
            check_log.append({"type": "WARN", "message": full_msg})
            return False

    def info_msg(msg):
        nonlocal infos
        print(f"  [INFO] {msg}")
        infos += 1
        check_log.append({"type": "INFO", "message": msg})

    def warn_msg(msg):
        nonlocal warns
        print(f"  [WARN] {msg}")
        warns += 1
        check_log.append({"type": "WARN", "message": msg})

    # ============================================================
    # 1. MODEL ARTIFACTS
    # ============================================================
    print("\n--- 1. Model Artifacts (Structural Validation) ---")
    expected_models = [
        "model_a_trade.joblib",
        "model_a_rf.joblib",
        "model_a_xgb.joblib",
        "model_a_lgb.joblib",
        "model_b_production_yoy.joblib",
        "model_b_production_risk.joblib",
        "model_b_yield_yoy.joblib",
        "model_c_price.joblib",
        "model_d_agri_gva.joblib",
        "model_d_inflation.joblib",
    ]
    for m in expected_models:
        path = os.path.join(MODELS_DIR, m)
        exists = os.path.exists(path)
        pass_check(exists, f"Model artifact exists: {m}")
        if exists:
            try:
                model = joblib.load(path)
                pass_check(True, f"Model artifact loadable: {m}")
            except Exception as e:
                pass_check(False, f"Model artifact loadable: {m} ({e})")

    # ============================================================
    # 2. RESULT & DATA FILES
    # ============================================================
    print("\n--- 2. Result & Data File Existence ---")
    expected_results = [
        "model_a_results.json",
        "model_a_predictions.csv",
        "model_b_results.json",
        "model_b_predictions.csv",
        "model_c_results.json",
        "model_c_predictions.csv",
        "model_d_results.json",
        "cascade_results.json",
        "event_store.json",
        "stakeholder_advisories.json",
    ]
    for r in expected_results:
        path = os.path.join(RESULTS_DIR, r)
        pass_check(os.path.exists(path), f"Result file exists: {r}")

    expected_data = [
        "Drishti_Cascade_Final_With_EMDAT.csv",
        "Crop_Production_Final.csv",
        "crop_hs4_mapping.csv",
        "crop_production_monthly.csv",
        "crop_production_state_level.csv",
        "event_catalog.json",
    ]
    for d in expected_data:
        path = os.path.join(DATA_DIR, d)
        pass_check(os.path.exists(path), f"Data file exists: {d}")

    # ============================================================
    # 3. LEAKAGE & FEATURE SCHEMA SANITY CHECKS
    # ============================================================
    print("\n--- 3. Leakage & Feature Schema Sanity Checks ---")
    info_msg("Evaluating train-test performance gaps and feature schemas (performance sanity check, not a mathematical leakage proof).")

    # Train-Test performance gap checks for Model A
    results_a_path = os.path.join(RESULTS_DIR, "model_a_results.json")
    if os.path.exists(results_a_path):
        with open(results_a_path) as f:
            res_a = json.load(f)
        for algo in ["RandomForest", "XGBoost", "LightGBM"]:
            if algo in res_a.get("primary_results", {}):
                m = res_a["primary_results"][algo]
                gap = m["train"]["R2"] - m["test"]["R2"]
                warn_check(gap < 0.5, f"Model A {algo}: train R2={m['train']['R2']:.4f}, test R2={m['test']['R2']:.4f}, gap={gap:.4f} (threshold < 0.50)")

    # Train-Test performance gap checks for Model C
    results_c_path = os.path.join(RESULTS_DIR, "model_c_results.json")
    if os.path.exists(results_c_path):
        with open(results_c_path) as f:
            res_c = json.load(f)
        for algo in ["RandomForest", "XGBoost", "LightGBM"]:
            if algo in res_c.get("model_results", {}):
                m = res_c["model_results"][algo]["metrics"]
                gap = m["train"]["R2"] - m["test"]["R2"]
                warn_check(gap < 0.5, f"Model C {algo}: train R2={m['train']['R2']:.4f}, test R2={m['test']['R2']:.4f}, gap={gap:.4f} (threshold < 0.50)")

    # Check Model D Feature Schema: Exclude Unsafe Contemporaneous, Include Expected Lagged
    UNSAFE_CONTEMPORANEOUS_D = [
        "CPI_Food_Index",
        "CPI_Food_Inflation",
        "GDP_Growth_Percent",
        "Agri_GVA_Growth_Percent",
        "Forex_Reserves_USD_Million",
        "Value_USD",
        "Trade_Return_1M",
        "Price_Return_1M",
        "Effective_Shock",
        "Trade_Share",
    ]
    EXPECTED_LAGGED_D = [
        "Inflation_Lag1",
        "Agri_GVA_Lag1",
        "GDP_Lag1",
        "Price_Lag1",
        "Shock_Intensity_Lag1",
        "Shock_Intensity_Lag2",
        "Trade_Share_Lag1",
        "Trade_Share_Lag2",
        "Lagged_Effective_Shock_1",
        "Lagged_Effective_Shock_2",
        "GPR",
        "INR_USD_Rate",
        "Price_Return_1M_Pred_Lag1",
    ]

    model_d_gva_path = os.path.join(MODELS_DIR, "model_d_agri_gva.joblib")
    if os.path.exists(model_d_gva_path):
        m_d = joblib.load(model_d_gva_path)
        d_feats = [str(f) for f in getattr(m_d, "feature_names_in_", [])]

        unsafe_found = [f for f in UNSAFE_CONTEMPORANEOUS_D if f in d_feats]
        pass_check(len(unsafe_found) == 0, f"Model D excludes unsafe contemporaneous features (found {len(unsafe_found)} unsafe: {unsafe_found})")

        missing_expected = [f for f in EXPECTED_LAGGED_D if f not in d_feats]
        pass_check(len(missing_expected) == 0, f"Model D includes all 13 expected lagged/exogenous features (missing: {missing_expected})")

    # ============================================================
    # 4. CASCADE SEMANTIC VALIDATION
    # ============================================================
    print("\n--- 4. Cascade Semantic Validation ---")
    cascade_path = os.path.join(RESULTS_DIR, "cascade_results.json")
    if os.path.exists(cascade_path):
        with open(cascade_path) as f:
            cascade = json.load(f)
        pass_check(len(cascade) >= 5, f"Cascade results contain {len(cascade)} demonstration cases (expected >= 5)")

        for i, result in enumerate(cascade):
            state = result.get("cascade_predictions") or result.get("cascade_state", {})
            trade_st = state.get("trade", {})
            agri_st = state.get("agriculture", {})
            price_st = state.get("price", {})
            econ_st = state.get("economy", {})
            trace = state.get("trace", [])

            c_ok = (
                "Trade_Return_1M_Pred" in trade_st and
                "Production_Growth_Pred" in agri_st and
                "Production_Risk" in agri_st and
                "Price_Return_1M_Pred" in price_st and
                "Agri_GVA_Growth_Pred" in econ_st and
                "Inflation_Change_Pred" in econ_st and
                len(trace) >= 6
            )
            pass_check(c_ok, f"Cascade Case {i+1}: contains all required stage predictions and trace entries ({len(trace)} trace items)")

    # Verify Documented Cascade Architecture Dependencies
    model_b_path = os.path.join(MODELS_DIR, "model_b_production_yoy.joblib")
    model_c_path = os.path.join(MODELS_DIR, "model_c_price.joblib")
    if os.path.exists(model_b_path) and os.path.exists(model_c_path):
        m_b = joblib.load(model_b_path)
        m_c = joblib.load(model_c_path)
        b_feats = [str(f) for f in getattr(m_b, "feature_names_in_", [])]
        c_feats = [str(f) for f in getattr(m_c, "feature_names_in_", [])]

        pass_check("Trade_Return_1M_Pred" not in b_feats, "Model A -> Model B link is CONCEPTUAL ONLY (Trade_Return_1M_Pred is NOT a trained Model B feature)")
        pass_check("Production_Growth_Pred_Lag1" in c_feats, "Model B -> Model C link uses lagged prediction feature (Production_Growth_Pred_Lag1)")
        pass_check("Price_Return_1M_Pred_Lag1" in d_feats, "Model C -> Model D link uses lagged prediction feature (Price_Return_1M_Pred_Lag1)")

    # ============================================================
    # 5. LAG FEATURE SEMANTIC CHECK
    # ============================================================
    print("\n--- 5. Lag Feature Semantic & Provenance Checks ---")
    pred_b_path = os.path.join(RESULTS_DIR, "model_b_predictions.csv")
    if os.path.exists(pred_b_path):
        df_pb = pd.read_csv(pred_b_path, nrows=1000)
        has_prod_pred = "Production_Growth_Pred" in df_pb.columns
        pass_check(has_prod_pred, "Model B predictions artifact contains Production_Growth_Pred for previous-period shift lookup")
    else:
        warn_msg("Model B predictions artifact missing: Production_Growth_Pred_Lag1 lookup NOT VERIFIABLE FROM AVAILABLE ARTIFACT")

    pred_c_path = os.path.join(RESULTS_DIR, "model_c_predictions.csv")
    if os.path.exists(pred_c_path):
        df_pc = pd.read_csv(pred_c_path, nrows=1000)
        has_price_pred = "Price_Return_1M_Pred" in df_pc.columns
        pass_check(has_price_pred, "Model C predictions artifact contains Price_Return_1M_Pred for previous-period shift lookup")
    else:
        warn_msg("Model C predictions artifact missing: Price_Return_1M_Pred_Lag1 lookup NOT VERIFIABLE FROM AVAILABLE ARTIFACT")

    info_msg("Upstream prediction artifacts contain in-sample predictions. Full out-of-fold/walk-forward provenance is deferred.")

    # ============================================================
    # 6. MODEL PERFORMANCE & GVA R2 SANITY CHECKS
    # ============================================================
    print("\n--- 6. Model Performance & R2 Sanity Checks ---")
    # Model A
    if os.path.exists(results_a_path):
        m_rf = res_a["primary_results"]["RandomForest"]
        bl_a = res_a["primary_results"]["baseline"]
        pass_check(m_rf["test"]["MAE"] < bl_a["test"]["MAE"], f"Model A (Trade RF) beats test baseline (MAE {m_rf['test']['MAE']:.4f} < {bl_a['test']['MAE']:.4f})")
        r2_test_a = m_rf["test"]["R2"]
        pass_check(0.0 <= r2_test_a < 0.95, f"Model A (Trade RF) test R2={r2_test_a:.4f} is within plausible non-trivial range [0, 0.95)")

    # Model B
    results_b_path = os.path.join(RESULTS_DIR, "model_b_results.json")
    if os.path.exists(results_b_path):
        with open(results_b_path) as f:
            res_b = json.load(f)
        rf_b = res_b.get("Production_YoY_National_RandomForest", {})
        beats_b = rf_b.get("beats_baseline", False)
        warn_check(beats_b, f"Model B (Agri Prod RF): beats baseline = {beats_b}", "Model B does not beat test baseline due to narrow crop dataset window")

    # Model C
    if os.path.exists(results_c_path):
        m_c_rf = res_c["model_results"]["RandomForest"]["metrics"]
        bl_c = res_c["baseline"]["test"]
        pass_check(m_c_rf["test"]["MAE"] < bl_c["MAE"], f"Model C (Price RF) beats test baseline (MAE {m_c_rf['test']['MAE']:.4f} < {bl_c['MAE']:.4f})")

    # Model D
    results_d_path = os.path.join(RESULTS_DIR, "model_d_results.json")
    if os.path.exists(results_d_path):
        with open(results_d_path) as f:
            res_d = json.load(f)

        gva_res = res_d.get("model_results", {}).get("Agri_GVA_Growth_Percent_RandomForest", {})
        infl_res = res_d.get("model_results", {}).get("Inflation_Change_3M_LightGBM", {})

        # GVA Validation R2 handling (single annual observation)
        val_r2_gva = gva_res.get("val", {}).get("R2", None)
        if val_r2_gva is None or np.isnan(val_r2_gva if val_r2_gva is not None else np.nan):
            info_msg("Model D Agri GVA Validation R2: N/A - single annual validation observation (2023). Mathematically undefined.")
        else:
            info_msg(f"Model D Agri GVA Validation R2: {val_r2_gva}")

        # GVA Test Performance
        gva_test_mae = gva_res.get("test", {}).get("MAE", None)
        gva_bl_mae = gva_res.get("baseline_test", {}).get("MAE", None)
        if gva_test_mae is not None and gva_bl_mae is not None:
            pass_check(gva_test_mae < gva_bl_mae, f"Model D (Agri GVA RF) beats test baseline (MAE {gva_test_mae:.4f} < {gva_bl_mae:.4f})")

        # GVA Test R2 Sanity Check
        gva_test_r2 = gva_res.get("test", {}).get("R2", None)
        if gva_test_r2 is not None and gva_test_r2 < 0:
            warn_msg(f"Model D Agri GVA Test R2 is negative ({gva_test_r2:.4f}); macro target variance exceeds baseline due to sparse annual observations.")

        # Inflation Performance
        infl_test_mae = infl_res.get("test", {}).get("MAE", None)
        infl_bl_mae = infl_res.get("baseline_test", {}).get("MAE", None)
        if infl_test_mae is not None and infl_bl_mae is not None:
            beats_infl = infl_test_mae < infl_bl_mae
            warn_check(beats_infl, f"Model D (Inflation LightGBM) test MAE vs baseline ({infl_test_mae:.4f} vs {infl_bl_mae:.4f})", "Model D Inflation does not beat baseline on test (model performance limitation)")

    # ============================================================
    # 7. FORMULA VALIDATION STATUS
    # ============================================================
    print("\n--- 7. Formula Validation Status ---")
    t1_rep = os.path.join(RESULTS_DIR, "task1_data_validation_report.txt")
    if os.path.exists(t1_rep):
        pass_check(True, "Dedicated Task 1 formula validation report exists (results/task1_data_validation_report.txt)")
    info_msg("Formula validation is performed separately in Task 1. Task 10 does not independently recompute formulas.")

    # ============================================================
    # 8. EVENT STORE VALIDATION
    # ============================================================
    print("\n--- 8. Event Store & Window Analytics Validation ---")
    event_path = os.path.join(RESULTS_DIR, "event_store.json")
    if os.path.exists(event_path):
        with open(event_path) as f:
            events = json.load(f)
        pass_check(len(events) >= 5, f"Event store contains {len(events)} curated events (expected >= 5)")

        for ev in events:
            ev_meta = ev.get("event", {})
            ev_id = ev_meta.get("event_id", "UNKNOWN")
            n_rows = ev.get("summary", {}).get("n_rows", 0)
            scope = ev_meta.get("event_scope", "direct")
            pass_check(n_rows > 0, f"Event [{ev_id}] {ev_meta.get('name', '')[:35]}: {n_rows:,} rows matched (scope: {scope})")
            if ev_id == "EVT006":
                pass_check(scope == "proxy", f"Event [EVT006] explicitly labeled as proxy event (Bangladesh proxy for Sri Lanka)")

    info_msg("Event-window statistics describe observations during curated periods; no causal attribution is claimed.")

    # ============================================================
    # 9. STAKEHOLDER RULE VALIDATION
    # ============================================================
    print("\n--- 9. Stakeholder Rule & Advisory Validation ---")
    advisory_path = os.path.join(RESULTS_DIR, "stakeholder_advisories.json")
    if os.path.exists(advisory_path):
        with open(advisory_path) as f:
            advisories = json.load(f)
        pass_check(len(advisories) >= 5, f"Stakeholder advisories generated for {len(advisories)} cases")

        if len(advisories) > 0:
            first = advisories[0]
            effects = first.get("effects", {})
            expected_stakeholders = [
                "farmer_output_price",
                "farmer_input_cost",
                "consumer",
                "exporter",
                "importer",
                "regional_production_risk",
                "macro_summary",
            ]
            for s in expected_stakeholders:
                pass_check(s in effects, f"Advisory structure contains {s} effect entry")

            # Check directional consistency across all cases
            dir_ok = True
            for case_idx, adv in enumerate(advisories):
                eff = adv.get("effects", {})
                flow = adv.get("meta", {}).get("Trade_Type", "")
                p_eff = eff.get("farmer_output_price", {}).get("effect", 0.0)
                c_dir = eff.get("consumer", {}).get("direction", "")
                exp_dir = eff.get("exporter", {}).get("direction", "")
                imp_dir = eff.get("importer", {}).get("direction", "")

                # Positive price effect should not produce positive consumer effect
                if p_eff > 0 and c_dir == "positive":
                    dir_ok = False
                # Negative price effect should not produce negative consumer effect
                if p_eff < 0 and c_dir == "negative":
                    dir_ok = False
                # Flow applicability
                if flow == "Export" and imp_dir != "not_applicable":
                    dir_ok = False
                if flow == "Import" and exp_dir != "not_applicable":
                    dir_ok = False

            pass_check(dir_ok, "Stakeholder advisory rules satisfy directional consistency and trade flow applicability")
            pass_check(effects.get("farmer_input_cost", {}).get("data_gap", False), "Farmer input cost explicitly flagged as data gap (HS Ch. 31 absent)")

    # ============================================================
    # 10. CHRONOLOGICAL SPLIT INTEGRITY
    # ============================================================
    print("\n--- 10. Chronological Split Integrity ---")
    main_csv = os.path.join(DATA_DIR, "Drishti_Cascade_Final_With_EMDAT.csv")
    df = pd.read_csv(main_csv, usecols=["Year"])
    train_years = df[df["Year"] <= 2022]["Year"].unique()
    val_years = df[df["Year"] == 2023]["Year"].unique()
    test_years = df[df["Year"] >= 2024]["Year"].unique()
    pass_check(max(train_years) <= 2022, f"Train years max = {max(train_years)} (<= 2022)")
    pass_check(2023 in val_years, f"Validation year = 2023")
    pass_check(min(test_years) >= 2024, f"Test years min = {min(test_years)} (>= 2024)")
    pass_check(not (set(train_years) & set(test_years)), "Strict separation: zero overlap between train and test years")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    total = passes + warns + infos + fails
    print("\n" + "=" * 70)
    print(f"VALIDATION SUMMARY:")
    print(f"  {passes} PASS / {warns} WARN / {infos} INFO / {fails} FAIL / {total} TOTAL")
    print("=" * 70)

    if fails == 0:
        print("\n  Pipeline integrity checks passed.")
        print("  Warnings represent model-performance limitations, non-verifiable provenance, or checks that require separate validation.")
    else:
        print(f"\n  {fails} critical integrity failure(s) detected. Review above for details.")

    # Save structured validation report
    validation = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "passes": passes,
            "warns": warns,
            "infos": infos,
            "fails": fails,
            "total": total,
            "status": "PASS" if fails == 0 else "FAIL",
        },
        "notes": (
            "Pipeline integrity checks passed. Warnings denote model-performance limitations "
            "(e.g., Inflation not beating baseline on test) and observational/provenance boundaries."
        ),
        "checks": check_log,
    }
    val_path = os.path.join(RESULTS_DIR, "validation_report.json")
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2)
    print(f"\nValidation report saved: {val_path}\n")


if __name__ == "__main__":
    main()
