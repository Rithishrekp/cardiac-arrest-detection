# Cardiac Arrest & Heart Risk Prediction System

**Internship Project — National Institute of Technology, Puducherry**

An ML-powered web application that predicts cardiac arrest / heart risk from ECG interval measurements using a trained XGBoost classifier.

---

## Problem Statement

Cardiac arrest and heart rhythm abnormalities are leading causes of mortality worldwide. Early detection of elevated cardiac risk from ECG interval data can help clinicians intervene before a critical event. This project provides a clean, professional dashboard where healthcare staff can enter patient ECG vitals and receive instant ML-based risk predictions.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React.js + Vite |
| Backend | FastAPI (Python) |
| Database | SQLite (local dev) / PostgreSQL (production) |
| ML | XGBoost, scikit-learn, pandas |

---

## Folder Structure

```
project-root/
├── frontend/               # React + Vite dashboard
│   ├── src/
│   │   ├── components/     # Layout, Navbar, RiskCard
│   │   ├── pages/          # Home, Predict, Result, History, About
│   │   ├── services/       # API client (api.js)
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── backend/                # FastAPI REST API
│   ├── app/
│   │   ├── main.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── routes/
│   │   │   ├── prediction.py
│   │   │   └── history.py
│   │   └── services/
│   │       └── prediction_service.py
│   ├── requirements.txt
│   └── .env
│
├── ml/                     # Existing ML pipeline (do not modify)
│   ├── saved_models/
│   │   └── best_xgboost.pkl
│   ├── inference/
│   └── ...
│
└── README.md
```

---

## ML Model Details

The model uses **18 ECG sliding-window features** derived from three raw interval inputs:

| Input | Description | Unit |
|-------|-------------|------|
| RR Interval | Time between consecutive R-peaks | ms |
| PP Interval | Time between consecutive P-waves | ms |
| QT Interval | Ventricular depolarization + repolarization | ms |

The backend computes rolling statistics (mean, variance, min, max, delta) over the last **5 readings per patient** before running inference.

**Risk categories:**

| Score | Level | Label |
|-------|-------|-------|
| 0–40% | Low | Normal |
| 41–70% | Medium | Primary Alert |
| 71–100% | High | Critical Emergency |

---

## How to Run

### Prerequisites

- Python 3.10+
- Node.js 18+
- pip and npm

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be available at **http://127.0.0.1:8000**

- Swagger docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard will be available at **http://localhost:5173**

The Vite dev server proxies `/api/*` requests to the backend automatically.

### 3. Run Backend Tests

```bash
cd backend
python test_backend.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | API and ML engine health check |
| `POST` | `/predict` | Submit patient data, get prediction |
| `GET` | `/history` | List all prediction records (`?search=`) |
| `GET` | `/history/stats` | Dashboard statistics |
| `GET` | `/history/{id}` | Single record with health suggestions |

### Sample Prediction Request

```json
{
  "patient_id": "P-1001",
  "patient_name": "Ramesh Kumar",
  "age": 45,
  "gender": "Male",
  "rr_interval": 1084.0,
  "pp_interval": 1090.0,
  "qt_interval": 448.0
}
```

---

## Environment Variables

### Backend (`backend/.env`)

```
DATABASE_URL=sqlite:///./cardiac_risk.db
PORT=8000
HOST=127.0.0.1
PREDICTION_MODEL=best_xgboost.pkl
```

For PostgreSQL:
```
DATABASE_URL=postgresql://user:password@localhost:5432/cardiac_risk
```

### Frontend (`frontend/.env`)

```
VITE_API_URL=/api
```

For production, set `VITE_API_URL=https://your-api-domain.com`

---

## Future Enhancements

- Real-time ECG waveform upload with CNN-BiLSTM ensemble
- User authentication and clinician role management
- Docker Compose deployment with PostgreSQL
- PDF report export for each assessment
- Live monitoring dashboard with WebSocket streaming
- Automated model retraining pipeline

---

## Medical Disclaimer

**This application provides ML-based cardiac risk predictions for research and educational purposes only.**

It is **NOT** a substitute for professional medical diagnosis, treatment, or advice. Always consult a qualified healthcare provider for medical decisions. In case of emergency, call your local emergency services immediately.

---

## License

Academic / internship project — NIT Puducherry.
