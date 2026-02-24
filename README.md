# FlowLab Console v0.1.0

Production-grade ML system for hourly traffic Level-of-Service prediction.

## Architecture

```
traffic_ml/
├── core.py            # Constants, data loading, drift detection (PSI/KS)
├── model.py           # Multi-output HistGBR + tuning + evaluation + explainability + versioning
├── predict_engine.py  # Autoregressive inference + Excel report builder
├── app.py             # Flask backend (REST API)
├── dashboard.html     # Web GUI dashboard
├── run.py             # CLI entry-point (preserves original predict.py behaviour)
├── requirements.txt
├── models/            # Versioned model artifacts (created on first train)
├── logs/              # Prediction log for rolling error monitoring
└── outputs/           # Generated Excel reports
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. CLI — exact original predict.py behaviour
```bash
# Place traffic_los_dataset.csv next to run.py, then:
python run.py                         # train + predict 2026-02-18 → Excel
python run.py --date 2026-03-05       # predict a different date
python run.py --tune --n-iter 30      # with hyperparameter tuning
```

### 3. Web application + REST API
```bash
python run.py --serve
# Open: http://localhost:5000/
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Train model with optional Optuna-style tuning |
| POST | `/predict` | Generate predictions + Excel report |
| GET  | `/explain` | Permutation feature importance |
| GET  | `/monitor` | Drift, rolling error, registry |
| GET  | `/versions` | List model versions |
| GET  | `/download/<file>` | Download Excel report |

### /train
```json
POST /train
{
  "tune": false,        // enable hyperparameter search
  "n_iter": 20,         // tuning candidates
  "n_splits": 5,        // time-series CV folds
  "retrain_full": true  // retrain on full data after evaluation
}
```

### /predict
```json
POST /predict
{
  "date": "2026-02-18",
  "version": "latest"
}
```

## Model Details

**Architecture:** `MultiOutputRegressor(HistGradientBoostingRegressor)` × 6 vehicle types  
- Trains a single shared-tuned model per-output, capturing inter-vehicle correlations through unified hyperparameter optimization
- Multi-output architecture allows joint tuning against the primary traffic objective

**Tuning Objective:** `0.7 × V/C MAE − 0.3 × LOS Accuracy`  
Optimized via `TimeSeriesSplit` cross-validation (no future leakage)

**Features (19 total):**
- Road encoding, hour, day-of-week, weekend flag, peak-hour flags, month
- Road infrastructure: lanes, design speed, base capacity, computed capacity
- Lag features: vol_total lag1-3, vc_ratio lag1-2, rolling means (3h, 6h)

**Sample weighting:** LOS F (V/C>1.0) gets 3× weight, LOS E 2×, LOS C/D 1.5×

**Structural constraints:**
- Jalan Pahang & Sultan Azlan Shah: motorcycle volume forced to 0
- Jalan Sultan Yahya Petra hours 0–9: all volumes forced to 0

## MLOps Guardrails

| Feature | Implementation |
|---------|---------------|
| Data drift | PSI (Population Stability Index) + KS test per feature |
| Prediction drift | Rolling V/C MAE over last N predictions, alert if > 0.15 |
| Feature integrity | Null rate checks, schema validation on load |
| Model versioning | Pickle + JSON metadata, symlink to latest |
| Explainability | Permutation importance (V/C MAE objective) |

## Excel Output

The Excel report is **fully backward-compatible** with the original `predict.py` format:
- One sheet per road (14 sheets total, same order)
- Identical merged cells, column widths, row heights
- Same formulas: PCU volume, V/C ratio, LOS grade (all Excel-native)
- Identical styling: title bar, parameter block, PCU table, LOS legend, data rows, TOTAL row
- Freeze panes at A12
