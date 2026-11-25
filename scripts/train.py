"""
Standalone training script — replicates the notebook pipeline and saves all model artifacts.
Works with real UNSW-NB15 CSVs (place in data/) or falls back to synthetic data for testing.

Usage:
    python scripts/train.py
    python scripts/train.py --samples 50000   # synthetic data size
"""

import os
import sys
import argparse

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, 'data')
MODEL_DIR = os.path.join(ROOT, 'models')


# ── Synthetic data ────────────────────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 10_000) -> pd.DataFrame:
    """
    Generate structured synthetic data with real signal so the model actually learns.
    Normal: low rate/bytes, UDP, INT state.  Attack: high rate/bytes, TCP, FIN/RST state.
    """
    rng = np.random.default_rng(42)
    n   = n_samples

    attack_cats = ['Generic', 'Exploits', 'Fuzzers', 'DoS',
                   'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms']

    labels    = rng.choice([0, 1], n, p=[0.40, 0.60])
    is_attack = labels == 1

    proto   = np.where(is_attack, rng.choice(['tcp','tcp','tcp','udp','icmp'], n),
                                  rng.choice(['udp','udp','udp','tcp','arp'], n))
    service = np.where(is_attack, rng.choice(['http','ftp','ssh','smtp','-'], n),
                                  rng.choice(['-','-','-','dns','http'], n))
    state   = np.where(is_attack, rng.choice(['FIN','CON','RST','FIN','FIN'], n),
                                  rng.choice(['INT','INT','INT','FIN','CON'], n))
    rate    = np.where(is_attack, rng.uniform(200_000, 2_000_000, n),
                                  rng.uniform(1_000, 50_000, n))
    sbytes  = np.where(is_attack, rng.integers(2000, 50_000, n),  rng.integers(100, 1_500, n))
    dbytes  = np.where(is_attack, rng.integers(1000, 30_000, n),  rng.integers(0, 500, n))
    spkts   = np.where(is_attack, rng.integers(10, 200, n),       rng.integers(1, 5, n))
    dpkts   = np.where(is_attack, rng.integers(5, 100, n),        rng.integers(0, 3, n))
    sttl    = np.where(is_attack, rng.choice([62,63,29,30,0,125], n),
                                  rng.choice([254,254,128,128,255], n))
    dttl    = np.where(is_attack, rng.choice([252,62,0,125], n),
                                  rng.choice([0,0,64,128], n))
    sload   = rate * rng.uniform(0.8, 1.2, n)
    dload   = np.where(is_attack, rate * rng.uniform(0.5, 1.0, n), np.zeros(n))
    sjit    = np.where(is_attack, rng.uniform(0, 0.0001, n),   rng.uniform(0.001, 0.05, n))
    djit    = np.where(is_attack, rng.uniform(0, 0.0001, n),   rng.uniform(0.001, 0.05, n))
    ct_srv  = np.where(is_attack, rng.integers(8, 20, n),  rng.integers(1, 4, n))
    ttl_vals = [0, 29, 30, 62, 63, 64, 125, 126, 127, 128, 254, 255]

    df = pd.DataFrame({
        'id':               range(1, n + 1),
        'dur':              np.where(is_attack, rng.uniform(0.001, 0.5, n), rng.uniform(0.000005, 0.0001, n)),
        'proto':            proto,
        'service':          service,
        'state':            state,
        'spkts':            spkts,
        'dpkts':            dpkts,
        'sbytes':           sbytes,
        'dbytes':           dbytes,
        'rate':             rate,
        'sttl':             sttl,
        'dttl':             dttl,
        'sload':            sload,
        'dload':            dload,
        'sloss':            rng.integers(0, 3, n),
        'dloss':            rng.integers(0, 3, n),
        'sinpkt':           rng.exponential(0.01, n),
        'dinpkt':           rng.exponential(0.01, n),
        'sjit':             sjit,
        'djit':             djit,
        'swin':             np.where(is_attack, rng.choice([255, 1024, 4096], n), rng.choice([0, 0, 255], n)),
        'stcpb':            rng.integers(0, 2**31, n),
        'dtcpb':            rng.integers(0, 2**31, n),
        'dwin':             np.where(is_attack, rng.choice([255, 1024, 4096], n), rng.choice([0, 0, 255], n)),
        'tcprtt':           rng.exponential(0.001, n),
        'synack':           rng.exponential(0.001, n),
        'ackdat':           rng.exponential(0.001, n),
        'smean':            np.where(is_attack, rng.integers(200, 1000, n), rng.integers(28, 300, n)),
        'dmean':            np.where(is_attack, rng.integers(100, 800, n),  rng.integers(0, 100, n)),
        'trans_depth':      np.where(is_attack, rng.integers(1, 8, n),      rng.integers(0, 1, n)),
        'response_body_len':np.where(is_attack, rng.integers(500, 10_000, n), rng.integers(0, 100, n)),
        'ct_srv_src':       ct_srv,
        'ct_state_ttl':     rng.integers(0, 6, n),
        'ct_dst_ltm':       np.where(is_attack, rng.integers(5, 20, n), rng.integers(1, 4, n)),
        'ct_src_dport_ltm': rng.integers(1, 10, n),
        'ct_dst_sport_ltm': rng.integers(1, 10, n),
        'ct_dst_src_ltm':   ct_srv,
        'is_ftp_login':     np.where(is_attack, rng.choice([0,1], n, p=[0.9,0.1]),
                                     rng.choice([0,1], n, p=[0.99,0.01])),
        'ct_ftp_cmd':       np.where(is_attack, rng.integers(0, 5, n), rng.integers(0, 1, n)),
        'ct_flw_http_mthd': np.where(is_attack, rng.integers(1, 5, n), rng.integers(0, 1, n)),
        'ct_src_ltm':       np.where(is_attack, rng.integers(5, 20, n), rng.integers(1, 4, n)),
        'ct_srv_dst':       ct_srv,
        'is_sm_ips_ports':  rng.choice([0, 1], n, p=[0.98, 0.02]),
        'attack_cat':       np.where(labels == 0, 'Normal', rng.choice(attack_cats, n)),
        'label':            labels,
    })
    return df


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(n_synthetic: int = 10_000) -> pd.DataFrame:
    train_csv = os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv')
    test_csv  = os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv')

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print("Loading real UNSW-NB15 dataset …")
        train = pd.read_csv(train_csv)
        test  = pd.read_csv(test_csv)
        df    = pd.concat([train, test]).reset_index(drop=True)
        print(f"  → {len(df):,} records loaded.")
        return df

    print("Dataset CSVs not found — generating synthetic data for testing.")
    print("  To use real data: place UNSW_NB15_training-set.csv and")
    print("  UNSW_NB15_testing-set.csv inside the data/ folder.")
    df = generate_synthetic_data(n_synthetic)
    print(f"  → {len(df):,} synthetic records generated.")
    return df


# ── Training pipeline ─────────────────────────────────────────────────────────

def train(n_synthetic: int = 10_000):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load
    df = load_data(n_synthetic)

    # 2. Clean
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop(columns=[c for c in ['id', 'attack_cat'] if c in df.columns])

    # 3. Encode categoricals — keep one encoder per column
    le_encoders = {}
    for col in df.select_dtypes(include=['str', 'object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_encoders[col] = le

    # 4. Split features / label
    X = df.drop('label', axis=1)
    y = df['label']
    feature_names = list(X.columns)

    # 5. Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Feature selection
    sel        = VarianceThreshold(threshold=0.01)
    X_selected = sel.fit_transform(X_scaled)
    selected_feature_names = [feature_names[i]
                               for i, keep in enumerate(sel.get_support()) if keep]
    print(f"Features kept by VarianceThreshold: {X_selected.shape[1]} / {len(feature_names)}")

    # 7. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42)

    # 8. Logistic Regression (baseline)
    print("\nTraining Logistic Regression …")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    print(f"  LR  Accuracy : {accuracy_score(y_test, lr_preds):.4f}")

    # 9. Random Forest
    print("Training Random Forest …")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    print(f"  RF  Accuracy : {accuracy_score(y_test, rf_preds):.4f}")
    print(f"  RF  F1-Score : {f1_score(y_test, rf_preds):.4f}")
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, rf_preds))

    # 10. Save all artifacts
    joblib.dump(rf_model,          os.path.join(MODEL_DIR, 'rf_model.pkl'))
    joblib.dump(lr_model,          os.path.join(MODEL_DIR, 'lr_model.pkl'))
    joblib.dump(scaler,            os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(sel,               os.path.join(MODEL_DIR, 'feature_selector.pkl'))
    joblib.dump(le_encoders,       os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    joblib.dump(feature_names,     os.path.join(MODEL_DIR, 'feature_names.pkl'))

    print(f"\nAll artifacts saved to {MODEL_DIR}/")

    # 11. Quick plots (saved as PNGs so they work headless)
    _save_plots(rf_model, lr_preds, rf_preds, y_test, X_selected, selected_feature_names)

    return rf_model, scaler, sel, le_encoders, feature_names


def _save_plots(rf_model, lr_preds, rf_preds, y_test,
                X_selected, selected_feature_names):
    fig_dir = os.path.join(ROOT, 'models')

    # Accuracy comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    models = ['Logistic Regression', 'Random Forest']
    scores = [accuracy_score(y_test, lr_preds), accuracy_score(y_test, rf_preds)]
    sns.barplot(x=models, y=scores, ax=ax)
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'accuracy_comparison.png'))
    plt.close(fig)

    # Feature importances
    importances = rf_model.feature_importances_
    top_idx     = np.argsort(importances)[-10:]
    top_names   = [selected_feature_names[i] if i < len(selected_feature_names)
                   else str(i) for i in top_idx]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_names, importances[top_idx])
    ax.set_title('Top 10 Feature Importances (Random Forest)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'feature_importances.png'))
    plt.close(fig)

    print("Plots saved: accuracy_comparison.png, feature_importances.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NIDS model on UNSW-NB15')
    parser.add_argument('--samples', type=int, default=10_000,
                        help='Synthetic dataset size when real CSVs are absent (default: 10000)')
    args = parser.parse_args()
    train(n_synthetic=args.samples)
