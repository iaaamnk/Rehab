# AI Rehabilitation Motion Analysis System

A full-stack AI-powered rehabilitation assistant that uses **MediaPipe Pose** for real-time webcam-based motion capture, a **Bidirectional LSTM** to detect compensatory movement patterns, and a **clinical scoring system** (ROM, Stability, Quality) delivered through a React dashboard backed by a FastAPI WebSocket server.

---

## ✨ Features

- **Real-Time Pose Tracking** — MediaPipe Pose via webcam, skeleton overlay rendered on canvas
- **LSTM Temporal Classifier** — Bidirectional LSTM trained on Kinect v2 data classifies every 30-frame window as healthy or compensatory
- **Clinical Scoring** — Range of Motion (ROM), Stability, and Movement Quality scores updated live, 0–100
- **Baseline Calibration** — First 30 frames establish a personalized posture baseline for relative lean detection
- **WebSocket Streaming** — Low-latency Python ↔ JS communication via FastAPI WebSocket
- **Real-Time Kinematics Chart** — Shoulder and trunk lean angles plotted live with Recharts
- **Clinical Feedback Alerts** — Concise, clinically-worded alerts when thresholds are exceeded

---

## 🏗️ Project Structure

```
mlproject/
├── backend/
│   ├── api/
│   │   ├── main.py          # FastAPI app, WebSocket stream, JWT auth
│   │   └── auth.py          # JWT helpers and OAuth2 router
│   ├── core/
│   │   ├── data_loader.py       # Kinect CSV loader with lowpass filter
│   │   ├── feature_extraction.py # Biomechanical features (Kinect + MediaPipe)
│   │   ├── rules_engine.py      # Clinical threshold rules + session scoring
│   │   ├── sequence_model.py    # BiLSTM model + sliding window buffer
│   │   └── database.py          # SQLite session logging
│   ├── models/
│   │   └── train.py         # Training script for the LSTM
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── index.css        # Design system & layout styles
│   │   └── main.jsx         # React entry point
│   ├── index.html           # MediaPipe CDN scripts loaded here
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   └── generate_mock_data.py
└── data_new/                # Kinect dataset (not committed — see Data section)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- A webcam

### 1 — Backend

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Copy env file and customise if needed
cp backend/.env.example backend/.env

# Run the API server (from project root)
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** If you have a trained model (`lstm_model.pth`), place it in `backend/models/`.  
> Without it the system falls back to rule-based detection only.

### 2 — Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

### 3 — Train the LSTM (optional)

1. Obtain the [KInAReT dataset](https://link.springer.com/article/10.1007/s11042-020-09781-1) and place it in `data_new/` using the structure:

   ```
   data_new/
   ├── H01/Rch_Fwr_Bck_L/Joint_Positions.csv
   ├── H01/Rch_Fwr_Bck_L/Labels.csv
   └── …
   ```

2. Run training:

   ```bash
   python -m backend.models.train
   ```

   This produces `backend/models/lstm_model.pth` and `backend/models/norm_params.npz`.

---

## 🧠 ML Architecture

| Component | Detail |
|:---|:---|
| **Input features** | 6 biomechanical signals per frame (shoulder angles, elbow angle, trunk sagittal lean, trunk lateral lean, shoulder height diff) |
| **Sequence length** | 30 frames (~1 s at 30 fps) |
| **Model** | Bidirectional LSTM — 2 layers, hidden size 64, dropout 0.3 |
| **Output** | Scalar probability → > 0.5 = compensatory movement |
| **Training data** | Kinect v2 captures — 10 healthy + 9 patient subjects |
| **Normalisation** | Per-feature z-score params saved to `norm_params.npz` |

---

## 🔐 Security Notes

- The default `SECRET_KEY` in `.env.example` **must** be replaced before any public deployment.
- The demo credential (`demo_doctor / password123`) in `database.py` is for local development only — remove or change it before deploying.
- The SQLite database file (`.db`) is excluded from the repository.

---

## 📦 Technology Stack

**Backend:** Python 3 · FastAPI · uvicorn · PyTorch · scikit-learn · SciPy · PyJWT · SQLite · bcrypt

**Frontend:** React 18 · Vite · Recharts · MediaPipe Pose (CDN)

---

## 🗂️ Data

The raw Kinect motion capture data is **not included** in this repository due to size and licensing constraints.

- The dataset used is based on the [KInAReT](https://link.springer.com/article/10.1007/s11042-020-09781-1) dataset structure.
- Pre-trained model weights are also excluded; run `backend/models/train.py` to generate them after obtaining the data.

---

## 📄 License

This project is released under the [MIT License](LICENSE).
