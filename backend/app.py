"""
FastAPI backend for the UNSW-NB15 Network IDS.

Start:
    uvicorn backend.app:app --reload           (from project root)
    uvicorn app:app --reload                   (from backend/ folder)

Endpoints:
    GET  /health          — liveness check
    POST /predict         — classify one network flow
    GET  /sample/normal   — example normal-traffic payload
    GET  /sample/attack   — example attack-traffic payload
    GET  /docs            — auto-generated Swagger UI
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
# Works whether uvicorn is run from project root or from backend/
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_HERE)
MODEL_DIR  = os.path.join(_ROOT, 'models')

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Network IDS API",
    description="UNSW-NB15 Network Intrusion Detection System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts ──────────────────────────────────────────────────────
MODEL_LOADED = False
rf_model = scaler = sel = le_encoders = feature_names = None

try:
    rf_model      = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
    scaler        = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    sel           = joblib.load(os.path.join(MODEL_DIR, 'feature_selector.pkl'))
    le_encoders   = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    MODEL_LOADED  = True
    print("[IDS] Models loaded successfully.")
except Exception as exc:
    print(f"[IDS] Warning: could not load models — {exc}")
    print("[IDS] Run  python scripts/train.py  to train and save models first.")


# ── Request schema ────────────────────────────────────────────────────────────
class NetworkFeatures(BaseModel):
    dur:               float = Field(0.000011, description="Connection duration (seconds)")
    proto:             str   = Field("udp",    description="Protocol (udp, tcp, icmp …)")
    service:           str   = Field("-",      description="Service (-,http,ftp,ssh …)")
    state:             str   = Field("INT",    description="Connection state (INT, FIN, CON …)")
    spkts:             int   = Field(2,        description="Source-to-dest packet count")
    dpkts:             int   = Field(0,        description="Dest-to-source packet count")
    sbytes:            int   = Field(496,      description="Source-to-dest bytes")
    dbytes:            int   = Field(0,        description="Dest-to-source bytes")
    rate:              float = Field(90909.0,  description="Packet rate (pps)")
    sttl:              int   = Field(254,      description="Source TTL")
    dttl:              int   = Field(0,        description="Dest TTL")
    sload:             float = Field(0.0,      description="Source bits/second load")
    dload:             float = Field(0.0,      description="Dest bits/second load")
    sloss:             int   = Field(0,        description="Source packet loss count")
    dloss:             int   = Field(0,        description="Dest packet loss count")
    sinpkt:            float = Field(0.0,      description="Source inter-packet arrival time (ms)")
    dinpkt:            float = Field(0.0,      description="Dest inter-packet arrival time (ms)")
    sjit:              float = Field(0.0,      description="Source jitter (ms)")
    djit:              float = Field(0.0,      description="Dest jitter (ms)")
    swin:              int   = Field(0,        description="Source TCP window size")
    stcpb:             int   = Field(0,        description="Source TCP base sequence number")
    dtcpb:             int   = Field(0,        description="Dest TCP base sequence number")
    dwin:              int   = Field(0,        description="Dest TCP window size")
    tcprtt:            float = Field(0.0,      description="TCP round-trip time (s)")
    synack:            float = Field(0.0,      description="SYN→SYN-ACK time (s)")
    ackdat:            float = Field(0.0,      description="SYN-ACK→ACK time (s)")
    smean:             int   = Field(248,      description="Mean source packet size")
    dmean:             int   = Field(0,        description="Mean dest packet size")
    trans_depth:       int   = Field(0,        description="HTTP pipeline depth")
    response_body_len: int   = Field(0,        description="HTTP response body length")
    ct_srv_src:        int   = Field(1,        description="# connections with same service+src in last 100")
    ct_state_ttl:      int   = Field(2,        description="# connections with same state+TTL in last 100")
    ct_dst_ltm:        int   = Field(1,        description="# connections to same dest in last 100")
    ct_src_dport_ltm:  int   = Field(1,        description="# connections from same src+dport in last 100")
    ct_dst_sport_ltm:  int   = Field(1,        description="# connections to same dst+sport in last 100")
    ct_dst_src_ltm:    int   = Field(2,        description="# connections between same src+dst in last 100")
    is_ftp_login:      int   = Field(0,        description="FTP login attempt (0/1)")
    ct_ftp_cmd:        int   = Field(0,        description="# FTP commands in this session")
    ct_flw_http_mthd:  int   = Field(0,        description="# HTTP methods in this flow")
    ct_src_ltm:        int   = Field(1,        description="# connections from same src in last 100")
    ct_srv_dst:        int   = Field(2,        description="# connections with same service+dst in last 100")
    is_sm_ips_ports:   int   = Field(0,        description="src/dst IPs and ports are equal (0/1)")

    class Config:
        json_schema_extra = {
            "example": {
                "dur": 0.000011, "proto": "udp", "service": "-", "state": "INT",
                "spkts": 2, "dpkts": 0, "sbytes": 496, "dbytes": 0,
                "rate": 90909.0, "sttl": 254, "dttl": 0, "sload": 0.0, "dload": 0.0,
                "sloss": 0, "dloss": 0, "sinpkt": 0.0, "dinpkt": 0.0,
                "sjit": 0.0, "djit": 0.0, "swin": 0, "stcpb": 0, "dtcpb": 0, "dwin": 0,
                "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 248, "dmean": 0,
                "trans_depth": 0, "response_body_len": 0, "ct_srv_src": 1,
                "ct_state_ttl": 2, "ct_dst_ltm": 1, "ct_src_dport_ltm": 1,
                "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 2, "is_ftp_login": 0,
                "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1,
                "ct_srv_dst": 2, "is_sm_ips_ports": 0,
            }
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess(features: NetworkFeatures) -> np.ndarray:
    df = pd.DataFrame([features.model_dump()])

    for col, le in le_encoders.items():
        if col in df.columns:
            known = set(le.classes_)
            df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])

    X        = df[feature_names]
    X_scaled = scaler.transform(X)
    X_sel    = sel.transform(X_scaled)
    return X_sel


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}


@app.post("/predict", tags=["Inference"])
def predict(features: NetworkFeatures):
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run  python scripts/train.py  from the project root."
        )

    X_sel = preprocess(features)
    pred  = int(rf_model.predict(X_sel)[0])
    prob  = rf_model.predict_proba(X_sel)[0]

    return {
        "prediction":    pred,
        "label":         "Attack" if pred == 1 else "Normal",
        "confidence":    round(float(max(prob)), 4),
        "probabilities": {
            "normal": round(float(prob[0]), 4),
            "attack": round(float(prob[1]), 4),
        },
    }


@app.get("/sample/normal", tags=["Samples"])
def sample_normal():
    """Return a sample normal-traffic payload ready to POST to /predict."""
    return {
        "dur": 0.000011, "proto": "udp", "service": "-", "state": "INT",
        "spkts": 2, "dpkts": 0, "sbytes": 496, "dbytes": 0,
        "rate": 90909.0, "sttl": 254, "dttl": 0, "sload": 0.0, "dload": 0.0,
        "sloss": 0, "dloss": 0, "sinpkt": 0.0, "dinpkt": 0.0,
        "sjit": 0.0, "djit": 0.0, "swin": 0, "stcpb": 0, "dtcpb": 0, "dwin": 0,
        "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 248, "dmean": 0,
        "trans_depth": 0, "response_body_len": 0, "ct_srv_src": 1,
        "ct_state_ttl": 2, "ct_dst_ltm": 1, "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 2, "is_ftp_login": 0,
        "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1,
        "ct_srv_dst": 2, "is_sm_ips_ports": 0,
    }


@app.get("/sample/attack", tags=["Samples"])
def sample_attack():
    """Return a sample attack-traffic payload ready to POST to /predict."""
    return {
        "dur": 0.021, "proto": "tcp", "service": "http", "state": "FIN",
        "spkts": 12, "dpkts": 8, "sbytes": 2860, "dbytes": 2320,
        "rate": 571428.57, "sttl": 62, "dttl": 252,
        "sload": 1_088_761.9, "dload": 882_285.71,
        "sloss": 0, "dloss": 0, "sinpkt": 1.909, "dinpkt": 2.625,
        "sjit": 0.0, "djit": 0.0, "swin": 255, "stcpb": 42, "dtcpb": 42, "dwin": 255,
        "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 238, "dmean": 290,
        "trans_depth": 2, "response_body_len": 1543, "ct_srv_src": 10,
        "ct_state_ttl": 2, "ct_dst_ltm": 10, "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 10, "is_ftp_login": 0,
        "ct_ftp_cmd": 0, "ct_flw_http_mthd": 2, "ct_src_ltm": 5,
        "ct_srv_dst": 10, "is_sm_ips_ports": 0,
    }
