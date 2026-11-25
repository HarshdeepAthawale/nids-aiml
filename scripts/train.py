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
    """Generate UNSW-NB15-shaped synthetic data for smoke-testing the pipeline."""
    rng = np.random.default_rng(42)

    proto_cats   = ['udp', 'tcp', 'arp', 'ospf', 'icmp', 'igmp', 'sctp', 'unas']
    service_cats = ['-', 'http', 'ftp', 'smtp', 'ssh', 'dns', 'ftp-data',
                    'irc', 'pop3', 'snmp', 'ssl', 'dhcp', 'radius']
    state_cats   = ['INT', 'FIN', 'CON', 'REQ', 'RST', 'URH', 'URN',
                    'PAR', 'ECO', 'ECR', 'TST', 'MAS', 'TXD', 'URP', 'no']
    attack_cats  = ['Generic', 'Exploits', 'Fuzzers', 'DoS',
                    'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms']

    n       = n_samples
    labels  = rng.choice([0, 1], n, p=[0.36, 0.64])
    ttl_vals = [0, 29, 30, 62, 63, 64, 125, 126, 127, 128, 254, 255]

    df = pd.DataFrame({
        'id':               range(1, n + 1),
        'dur':              rng.exponential(0.1, n),
        'proto':            rng.choice(proto_cats, n),
        'service':          rng.choice(service_cats, n),
        'state':            rng.choice(state_cats, n),
        'spkts':            rng.integers(1, 100, n),
        'dpkts':            rng.integers(0, 100, n),
        'sbytes':           rng.integers(100, 10_000, n),
        'dbytes':           rng.integers(0, 10_000, n),
        'rate':             rng.exponential(100_000, n),
        'sttl':             rng.choice(ttl_vals, n),
        'dttl':             rng.choice(ttl_vals, n),
        'sload':            rng.exponential(1e6, n),
        'dload':            rng.exponential(1e6, n),
        'sloss':            rng.integers(0, 10, n),
        'dloss':            rng.integers(0, 10, n),
        'sinpkt':           rng.exponential(0.01, n),
        'dinpkt':           rng.exponential(0.01, n),
        'sjit':             rng.exponential(0.001, n),
        'djit':             rng.exponential(0.001, n),
        'swin':             rng.choice([0, 255, 1024, 2048, 4096, 8192, 65535], n),
        'stcpb':            rng.integers(0, 2**31, n),
        'dtcpb':            rng.integers(0, 2**31, n),
        'dwin':             rng.choice([0, 255, 1024, 2048, 4096, 8192, 65535], n),
        'tcprtt':           rng.exponential(0.001, n),
        'synack':           rng.exponential(0.001, n),
        'ackdat':           rng.exponential(0.001, n),
        'smean':            rng.integers(28, 2000, n),
        'dmean':            rng.integers(0, 2000, n),
        'trans_depth':      rng.integers(0, 10, n),
        'response_body_len':rng.integers(0, 10_000, n),
        'ct_srv_src':       rng.integers(1, 20, n),
        'ct_state_ttl':     rng.integers(0, 6, n),
        'ct_dst_ltm':       rng.integers(1, 20, n),
        'ct_src_dport_ltm': rng.integers(1, 20, n),
        'ct_dst_sport_ltm': rng.integers(1, 20, n),
        'ct_dst_src_ltm':   rng.integers(1, 20, n),
        'is_ftp_login':     rng.choice([0, 1], n, p=[0.98, 0.02]),
        'ct_ftp_cmd':       rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.95, 0.02, 0.01, 0.01, 0.005, 0.005]),
        'ct_flw_http_mthd': rng.choice([0, 1, 2, 3, 4], n, p=[0.9, 0.05, 0.03, 0.01, 0.01]),
        'ct_src_ltm':       rng.integers(1, 20, n),
        'ct_srv_dst':       rng.integers(1, 20, n),
        'is_sm_ips_ports':  rng.choice([0, 1], n, p=[0.99, 0.01]),
        'attack_cat':       np.where(labels == 0, 'Normal',
                                     rng.choice(attack_cats, n)),
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
