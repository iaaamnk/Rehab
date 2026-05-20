"""
FastAPI backend — WebSocket streaming with LSTM inference, JWT auth, and
session persistence.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
import uvicorn
import os, sys, json
import numpy as np
import jwt as pyjwt
import torch
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.feature_extraction import KinematicFeatureExtractor, FEATURE_NAMES
from core.rules_engine import RulesEngine
from core.sequence_model import LSTMCompensationDetector, SequenceBuffer
from core.database import verify_user, create_user, log_session, get_user_sessions

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Live Rehabilitation Motion API")

SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE-ME-IN-PRODUCTION")
ALGORITHM  = "HS256"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load LSTM model ───────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pth")
NORM_PATH  = os.path.join(MODEL_DIR, "norm_params.npz")

lstm_model  = None
norm_mean   = None
norm_std    = None

if os.path.exists(MODEL_PATH) and os.path.exists(NORM_PATH):
    try:
        lstm_model = LSTMCompensationDetector(input_size=len(FEATURE_NAMES))
        lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        lstm_model.eval()
        _np = np.load(NORM_PATH)
        # Pre-convert to float32 tensors once at startup — avoids per-frame conversion
        norm_mean = torch.tensor(_np["mean"], dtype=torch.float32)
        norm_std  = torch.tensor(_np["std"],  dtype=torch.float32)
        print("✓  LSTM model loaded")
    except Exception as e:
        print(f"✗  LSTM load failed: {e}")
        lstm_model = None
else:
    print("⚠  LSTM model not found — run  python models/train.py  first")


# ── JWT helpers ───────────────────────────────────────────────────────
def _create_token(user_id: int, username: str) -> str:
    return pyjwt.encode(
        {"sub": str(user_id), "username": username,
         "exp": datetime.utcnow() + timedelta(hours=24)},
        SECRET_KEY, algorithm=ALGORITHM,
    )

def _verify_token(token: str) -> dict | None:
    try:
        return pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except (pyjwt.ExpiredSignatureError, pyjwt.InvalidTokenError):
        return None


# ── REST endpoints ────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Rehabilitation API is ready",
            "lstm_available": lstm_model is not None}


@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    uid = verify_user(form_data.username, form_data.password)
    if not uid:
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": _create_token(uid, form_data.username),
            "token_type": "bearer", "user_id": uid}


@app.post("/auth/register")
async def register(username: str, password: str):
    if not create_user(username, password):
        raise HTTPException(400, "Username already exists")
    return {"message": "User created"}


@app.get("/sessions/{user_id}")
async def get_sessions(user_id: int):
    return {"sessions": get_user_sessions(user_id)}


# ── WebSocket streaming ──────────────────────────────────────────────
class _Session:
    def __init__(self):
        self.extractor = KinematicFeatureExtractor(calibration_frames=30)
        self.rules     = RulesEngine()
        self.seq_buf   = SequenceBuffer(seq_length=30, feature_names=FEATURE_NAMES)
        self.user_id: int | None = None


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    sess = _Session()

    # ── optional auth handshake ───────────────────────────────────────
    try:
        first = json.loads(await ws.receive_text())
        if "token" in first:
            payload = _verify_token(first["token"])
            if payload:
                sess.user_id = int(payload["sub"])
                await ws.send_text(json.dumps(
                    {"type": "auth", "status": "ok",
                     "username": payload["username"]}))
            else:
                await ws.send_text(json.dumps(
                    {"type": "auth", "status": "ok", "username": "guest"}))
        else:
            await ws.send_text(json.dumps(
                {"type": "auth", "status": "ok", "username": "guest"}))
            # If first msg already contained landmarks, process below
            if "landmarks" in first:
                _process_frame(sess, first)
    except Exception:
        await ws.send_text(json.dumps(
            {"type": "auth", "status": "ok", "username": "guest"}))

    print(f"Session started  user_id={sess.user_id}")

    # ── main loop ─────────────────────────────────────────────────────
    try:
        while True:
            data = json.loads(await ws.receive_text())
            if "landmarks" not in data:
                continue

            resp = _process_frame(sess, data)
            if resp:
                await ws.send_text(json.dumps(resp))

    except WebSocketDisconnect:
        _save_session(sess)
        print("Client disconnected")
    except Exception as e:
        _save_session(sess)
        print(f"WS error: {e}")


def _process_frame(sess: _Session, data: dict) -> dict | None:
    features = sess.extractor.process_frame(data["landmarks"])
    if not features:
        return None

    feedback = sess.rules.evaluate_frame(features)

    # LSTM prediction
    ml_comp = False
    ml_conf = 0.0
    if lstm_model is not None:
        sess.seq_buf.add_frame(features)
        if sess.seq_buf.is_ready():
            seq = sess.seq_buf.get_sequence()
            if seq is not None and norm_mean is not None:
                seq_norm = (seq - norm_mean) / (norm_std + 1e-8)
                with torch.no_grad():
                    logits  = lstm_model(seq_norm)
                    ml_conf = float(torch.sigmoid(logits).item())
                    ml_comp = ml_conf > 0.5

    scores = sess.rules.get_session_scores()
    overall_feedback = sess.rules.get_overall_feedback()

    return {
        "type": "frame",
        "features": features,
        "is_compensatory": feedback["is_compensatory_rule_based"] or ml_comp,
        "ml_confidence": ml_conf,
        "is_calibrated": sess.extractor.is_calibrated,
        "scores": scores,
        "overall_feedback": overall_feedback,
    }


def _save_session(sess: _Session):
    scores = sess.rules.get_session_scores()
    if sess.user_id and scores["total_score"] > 0:
        log_session(
            sess.user_id,
            datetime.now().isoformat(),
            scores["rom_score"],
            scores["stability_score"],
            scores["quality_score"],
            scores["total_score"],
        )
        print(f"Session saved  user={sess.user_id}  scores={scores}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
