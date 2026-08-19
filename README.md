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

## Repository Structure

```
├── data/                          # Raw & processed data (CSVs)
├── models/                        # Saved trained model artifacts
├── results/                       # Metrics, feature importances, predictions
├── scripts/
│   ├── task1_data_validation.py   # Data validation & formula verification
│   ├── train_model_a_trade.py     # Model A: Trade Impact
│   ├── train_model_b_agriculture.py  # Model B: Agricultural Impact
│   ├── train_model_c_price.py     # Model C: Price Impact
│   ├── train_model_d_economy.py   # Model D: Economic Impact
│   ├── cascade_orchestrator.py    # Full cascade (Digital Twin)
│   ├── stakeholder_engine.py      # Rules-based stakeholder disaggregation
│   └── agents/                    # LLM-based explanation agents
├── requirements.txt
└── README.md
```

## Data

Two final processed datasets:
- `Drishti_Cascade_Final_With_EMDAT.csv` — 139,626 rows × 86 columns (2018–2025, 15 partner countries, 168 HS4 codes)
- `Crop_Production_Final.csv` — 80,450 rows × 24 columns (2018-19 to 2022-23, 37 states, 736 districts, 54 crops)

## Reproducibility

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place both CSV files in the `data/` directory.

3. Run scripts sequentially:
   ```bash
   python scripts/task1_data_validation.py
   python scripts/train_model_a_trade.py
   python scripts/train_model_b_agriculture.py
   python scripts/train_model_c_price.py
   python scripts/train_model_d_economy.py
   python scripts/cascade_orchestrator.py
   ```

**Global random seed**: `RANDOM_STATE = 42` used in all scripts.

## Chronological Data Splits

| Split | Period | Purpose |
|-------|--------|---------|
| Train | ≤ 2022 | Model training |
| Validation | 2023 | Hyperparameter tuning |
| Test | ≥ 2024 | Final evaluation |

Model B uses a narrower window (through 2022-23) due to crop data availability.

## Key Methodological Notes

- **No random splits** — all splits are chronological for time-series integrity.
- **No same-period leakage** — all macro/target-adjacent features are lagged before use.
- **Every model is benchmarked** against a naive persistence baseline.
- **Suspiciously high R²** (>0.95) triggers a mandatory feature re-audit.
- **Stakeholder disaggregation** is rules-based (no labeled ground truth exists).
