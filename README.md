# Drishti — Geopolitical Shock Cascade Model for Indian Agriculture

## Overview

Drishti models how geopolitical shocks propagate through India's agricultural economy via a sequential cascade:

```
Geopolitical Event → Trade Impact → Agricultural Impact → Price Impact → Agri-Economic Impact
                                                                                ↓
                                                          Stakeholder-Disaggregated Advisory
```

The core engineered concept is **exposure-aware shock weighting**: a geopolitical event's economic relevance to India is proportional to the trade share of the affected commodity with the affected partner country.

**Contribution**: This exact cascade combination applied end-to-end to India's domestic agricultural economy with disaggregated stakeholder output. Individual components (exposure-weighted indices, GDELT-based geopolitical indices, cascade prediction, LLM-agent explainability) have prior published work — this project's contribution is the applied system integrating them.

---

## Canonical Pipeline & Walk-Forward OOF Architecture

To guarantee strict temporal validity and eliminate in-sample leakage throughout the multi-stage cascade, all upstream predictions are generated via **expanding-window chronological walk-forward out-of-fold (OOF)** estimation:
- Prediction for year $Y$ strictly uses models trained on data $\le Y-1$ ($\text{Training\_End\_Year} < \text{Prediction\_Year}$).
- The 2018 cold-start period is explicitly preserved as unmapped/unavailable.
- Downstream models (Model C and Model D) ingest verified OOF predictions and previous-period $t-1$ lags.
- Hyperparameter tuning and model selection are conducted **strictly on Train ($\le 2022$) and Validation ($2023$)**.
- The final Test window ($\ge 2024$) remains frozen and untouched during all tuning decisions.

---

## Repository Structure

```
├── data/                             # Raw & processed datasets and event catalog
│   ├── Drishti_Cascade_Final_With_EMDAT.csv
│   ├── Crop_Production_Final.csv
│   └── event_catalog.json
├── models/                           # Saved trained canonical model artifacts
│   ├── model_a_trade.joblib          # Model A: Trade Impact
│   ├── model_b_production_yoy.joblib # Model B: Production YoY
│   ├── model_b_production_risk.joblib# Model B: Production Risk Classifier
│   ├── model_c_price.joblib          # Model C: Price Impact (OOF-trained)
│   ├── model_d_agri_gva.joblib       # Model D: Agri GVA (OOF-trained)
│   └── model_d_inflation.joblib      # Model D: Food Inflation (OOF-trained)
├── results/                          # Validated OOF predictions, metrics & audit reports
│   ├── model_a_predictions_oof.csv
│   ├── model_b_predictions_oof.csv
│   ├── model_c_predictions_oof.csv
│   ├── cascade_results.json
│   ├── event_store.json
│   ├── stakeholder_advisories.json
│   ├── validation_report.json
│   └── phase2_provenance_audit.json
├── scripts/                          # Canonical implementation scripts
│   ├── task1_data_validation.py      # Task 1: Dataset & formula validation
│   ├── train_model_a_trade.py        # Task 2: Model A (Trade Impact)
│   ├── train_model_b_agriculture.py  # Task 3: Model B (Agricultural Impact)
│   ├── generate_oof_predictions.py   # Expanding-window OOF generator for Models A & B
│   ├── train_model_c_price.py        # Task 4: Model C (Price Impact OOF)
│   ├── train_model_d_economy.py      # Tasks 5+6: Model D (Economic Impact OOF)
│   ├── cascade_orchestrator.py       # Task 7: Cascade Orchestrator (Digital Twin)
│   ├── stakeholder_engine.py         # Task 8: Rules-based Stakeholder Engine
│   ├── event_store.py                # Task 9: Curated Event Catalog & Window Analytics
│   ├── validate_pipeline.py          # Task 10: End-to-End Pipeline Validation Suite
│   └── audit_phase2_oof.py           # OOF Provenance & Temporal Audit Benchmark
├── requirements.txt
└── README.md
```

---

## Reproducibility & Pipeline Execution

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place `Drishti_Cascade_Final_With_EMDAT.csv` and `Crop_Production_Final.csv` in `data/`.

3. Run the canonical sequence:
   ```bash
   # 1. Validate dataset and derived lag formulas
   python scripts/task1_data_validation.py

   # 2. Train baseline models & generate upstream walk-forward OOF predictions
   python scripts/train_model_a_trade.py
   python scripts/train_model_b_agriculture.py
   python scripts/generate_oof_predictions.py

   # 3. Retrain downstream models using temporally valid OOF inputs
   python scripts/train_model_c_price.py
   python scripts/train_model_d_economy.py

   # 4. Run full cascade orchestrator & downstream modules
   python scripts/cascade_orchestrator.py
   python scripts/stakeholder_engine.py
   python scripts/event_store.py

   # 5. Run end-to-end pipeline integrity validation
   python scripts/validate_pipeline.py
   ```

**Global Random Seed**: `RANDOM_STATE = 42` used consistently across all scripts.

---

## Chronological Data Splits

| Split | Period | Purpose |
|---|---|---|
| **Train** | $\le 2022$ | Model training ($2018$ cold-start for OOF) |
| **Validation** | $2023$ | Hyperparameter optimization & model selection |
| **Test** | $\ge 2024$ | Frozen out-of-sample evaluation |

---

## Key Methodological Principles

- **Walk-Forward OOF Provenance**: All cascade links strictly use out-of-fold predictions with $\text{Training\_End\_Year} < \text{Prediction\_Year}$.
- **No Same-Period Leakage**: All upstream cascade features and target-adjacent indicators are strictly lagged prior to inclusion.
- **Honest Benchmarking**: Every model is benchmarked against naive persistence baselines.
- **Data Gap Transparency**: Fertilizer input costs (HS Ch. 31 absent) and unmapped non-crop agricultural periods are explicitly reported as data gaps rather than fabricated.
- **Observational Event Analytics**: Event-window analytics describe historical conditions without making unverified causal claims.
- **Rules-Based Stakeholder Disaggregation**: Stakeholder advisories use deterministic, domain-grounded rules based on trade flow direction and price transmission.
