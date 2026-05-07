# 📈 microGCC - Time Series Forecasting System

A production-grade, end-to-end sales forecasting backend built with **FastAPI**, **Python 3.12+**, and four powerful ML/DL models. The system trains per-state models on historical weekly sales data, automatically selects the best model, and exposes predictions through a RESTful API.

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  FastAPI      │────▶│  Services    │────▶│  Pipelines       │
│  REST API     │     │  (cache,     │     │  (preprocessing, │
│  (Swagger UI) │     │   orchestr.) │     │   training,      │
└──────────────┘     └──────────────┘     │   forecasting)   │
                                           └────────┬─────────┘
                                                    │
                              ┌──────────────────────┼──────────────────────┐
                              ▼                      ▼                      ▼
                     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
                     │  Models      │     │  Features    │     │  Data        │
                     │  ARIMA       │     │  Lags        │     │  Excel       │
                     │  Prophet     │     │  Rolling     │     │  Cleaning    │
                     │  XGBoost     │     │  Calendar    │     │  Resampling  │
                     │  LSTM        │     │  Holidays    │     │              │
                     └──────────────┘     └──────────────┘     └──────────────┘
```

## ✨ Key Features

| Feature | Description |
|---|---|
| **4 Forecasting Models** | SARIMAX, Prophet, XGBoost, LSTM |
| **Auto Model Selection** | Best model picked per state via RMSE |
| **Recursive Forecasting** | XGBoost & LSTM use autoregressive multi-step prediction |
| **Model Versioning** | `saved_models/{state}/v1/`, `v2/`, … |
| **Config-Driven** | `config.yaml` + `.env` for all hyperparameters |
| **Background Training** | Non-blocking `POST /train` via FastAPI BackgroundTasks |
| **Parallel Training** | `concurrent.futures` for multi-state training |
| **Prediction Caching** | TTL-based cache avoids redundant computation |
| **Confidence Intervals** | Lower/upper bounds on every forecast |
| **Centralized Metrics** | `metrics/all_metrics.csv` registry |
| **Reports & Charts** | Auto-generated model comparison and distribution plots |
| **Pydantic Schemas** | Typed request/response contracts |
| **Docker Ready** | Dockerfile + docker-compose.yml |

---

## 📂 Project Structure

```
forecasting-system/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── forecast.py       # GET /forecast/{state}
│   │   │   ├── train.py          # POST /train, GET /train/status
│   │   │   └── metrics.py        # GET /metrics, GET /metrics/{state}
│   │   └── main.py               # FastAPI app + health endpoint
│   ├── core/
│   │   ├── config.py             # Pydantic Settings + YAML loader
│   │   ├── logger.py             # Centralized logging
│   │   └── constants.py          # Column names, model registry
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── dataset.xlsx
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── base_model.py         # ABC interface
│   │   ├── arima_model.py        # SARIMAX
│   │   ├── prophet_model.py      # Facebook Prophet
│   │   ├── xgboost_model.py      # XGBoost + recursive forecast
│   │   ├── lstm_model.py         # LSTM + MinMaxScaler
│   │   └── model_selector.py     # Train → Evaluate → Select
│   ├── pipelines/
│   │   ├── preprocessing.py      # Load, clean, resample
│   │   ├── training_pipeline.py  # End-to-end training
│   │   └── forecasting_pipeline.py
│   ├── schemas/
│   │   └── schemas.py            # Pydantic models
│   ├── services/
│   │   ├── train_service.py
│   │   ├── forecast_service.py   # With caching
│   │   └── metrics_service.py
│   ├── utils/
│   │   ├── metrics.py            # RMSE, MAE, MAPE
│   │   ├── helpers.py            # I/O, versioning, timing
│   │   └── validators.py         # Data validation
│   └── saved_models/
├── metadata/                     # Per-state model metadata JSON
├── metrics/                      # Centralized all_metrics.csv
├── reports/                      # Auto-generated charts
├── tests/
├── config.yaml
├── .env
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.py
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
# Clone / navigate to the project
cd forecasting-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python run.py
```

The server starts at **http://localhost:8000**.

---

## 📡 API Usage

### Swagger UI

Open **http://localhost:8000/docs** in your browser for interactive API documentation.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + diagnostics |
| `POST` | `/train` | Trigger background model training |
| `GET` | `/train/status` | Check training progress |
| `GET` | `/forecast/{state}` | Get 8-week forecast for a state |
| `GET` | `/metrics` | All model comparison metrics |
| `GET` | `/metrics/{state}` | Metrics for a specific state |

### Example: Get Forecast

```bash
curl http://localhost:8000/forecast/California
```

```json
{
  "state": "California",
  "best_model": "xgboost",
  "forecast_horizon_weeks": 8,
  "forecast": [
    {
      "date": "2023-12-10",
      "predicted_sales": 445123.45,
      "lower_bound": 400611.11,
      "upper_bound": 489635.80
    }
  ]
}
```

---

## 🤖 Models

### 1. SARIMAX (ARIMA)
- Statistical model with trend + seasonal components
- Automatic fallback to simpler order on convergence failure
- Native confidence intervals

### 2. Facebook Prophet
- Handles trend, seasonality, and US holidays
- Weekly frequency forecasting
- Built-in uncertainty intervals

### 3. XGBoost
- Gradient-boosted trees on engineered features
- **Recursive forecasting**: predicts t+1, appends, predicts t+2, …
- Feature importance analysis

### 4. LSTM (Deep Learning)
- Sequence-to-one architecture with dropout
- MinMaxScaler normalization
- Sliding window (12-step) input
- Recursive multi-step forecasting

---

## 🔧 Configuration

### `config.yaml`
All hyperparameters, feature settings, and model configs.

### `.env`
Environment variables for paths, forecast horizon, server settings.

---

## 🐳 Docker

```bash
# Build and run
docker-compose up --build

# Or just Docker
docker build -t forecasting-system .
docker run -p 8000:8000 forecasting-system
```

---

## ☁️ Deployment

### Option 1: Render (PaaS)
1. Fork or push this repository to GitHub.
2. Go to [Render](https://render.com/) and create a new **Web Service**.
3. Connect your GitHub repository.
4. Set the following:
   - **Environment:** `Docker`
   - **Build Command:** (leave blank, Render uses the Dockerfile)
   - **Start Command:** (leave blank)
5. Add any required environment variables from your `.env` file in the Render dashboard.
6. Click **Deploy**.

### Option 2: AWS / DigitalOcean (VPS)
1. SSH into your instance.
2. Clone the repository: `git clone https://github.com/AyushPrakash414/microGCC.git`
3. Navigate to the project directory: `cd microGCC/forecasting-system`
4. Install Docker and Docker Compose on your server.
5. Run the system in detached mode: `docker-compose up -d --build`
6. Expose port `8000` in your server's firewall/security group settings.

```bash
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📊 Reports

After training (`POST /train`), check the `reports/` directory for:

- `model_distribution.png` — Pie chart of best model selection
- `rmse_comparison.png` — Bar chart comparing RMSE across states
- `training_time.png` — Training duration by model

---

## 📋 Forecasting Workflow

```
1. POST /train
   └── Preprocess data (clean, resample weekly)
   └── Engineer features (lags, rolling, calendar, holidays)
   └── For each state:
       └── Chronological train/validation split
       └── Train ARIMA, Prophet, XGBoost, LSTM
       └── Evaluate on validation set
       └── Select best model (lowest RMSE)
       └── Save models (versioned) + metadata
   └── Generate comparison reports

2. GET /forecast/{state}
   └── Load best model from metadata
   └── Predict next 8 weeks
   └── Return with confidence intervals
```

---

## 📝 License

MIT
