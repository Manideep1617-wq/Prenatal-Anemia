"""
Trains a clean XGBoost model on CBC-only data (no maternal mixing) and saves it.
This replaces the broken combined-dataset model.
Run once locally:  python retrain_xgb_cbc.py
"""
import os, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

DATA_DIR  = r'C:\Users\gshan\OneDrive\Documents\PrenatalAnemia\data'
SAVE_PATH = r'C:\Users\gshan\OneDrive\Documents\PrenatalAnemia\results\checkpoints\xgb_tabular.json'
SCALER_PATH = r'C:\Users\gshan\OneDrive\Documents\PrenatalAnemia\results\checkpoints\scaler_params.json'

# ── 1. Load CBC data ──────────────────────────────────────────────────────────
print("Loading CBC dataset...")
cbc = pd.read_csv(os.path.join(DATA_DIR, 'cbc_dataframe.csv'))
cbc = cbc[cbc['HGB'].notna()].copy()
cbc['label'] = (cbc['HGB'] < 11.0).astype(int)

CBC_FEATURES = ['WBC','RBC','HGB','HCT','MCV','MCHC','MCH','RDW','PLT','MPV']
keep = [c for c in CBC_FEATURES if c in cbc.columns] + ['label']
cbc = cbc[keep].dropna()
print(f"  Rows: {len(cbc):,}  |  Anemic: {cbc['label'].sum():,}  |  Non-Anemic: {(cbc['label']==0).sum():,}")

# ── 2. Also load maternal data for the remaining 7 features ──────────────────
# We'll train on CBC features only (10 features) — the most reliable signals
# Maternal vitals (Age, BMI, etc.) don't directly indicate anemia

FEATURES = sorted(CBC_FEATURES)  # alphabetical
print(f"Features ({len(FEATURES)}): {FEATURES}")

X = cbc[FEATURES].values.astype(np.float32)
y = cbc['label'].values.astype(int)

# ── 3. Save scaler params (for inference) ────────────────────────────────────
mean = X.mean(axis=0)
std  = X.std(axis=0) + 1e-8
scaler_params = {feat: {'mean': float(mean[i]), 'std': float(std[i])}
                 for i, feat in enumerate(FEATURES)}
with open(SCALER_PATH, 'w') as f:
    json.dump({'features': FEATURES, 'mean': mean.tolist(), 'std': std.tolist(),
               'params': scaler_params}, f, indent=2)
print(f"Scaler params saved -> {SCALER_PATH}")

# Scale
X_scaled = (X - mean) / std

# ── 4. Train/val split ────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2,
                                              random_state=42, stratify=y)
print(f"Train: {len(X_tr):,}  |  Val: {len(X_val):,}")

# ── 5. Train XGBoost ─────────────────────────────────────────────────────────
scale_pos = int((y_tr == 0).sum()) / int((y_tr == 1).sum())
print(f"scale_pos_weight = {scale_pos:.1f}")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=1,
)
model.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          verbose=50)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]
acc  = accuracy_score(y_val, y_pred)
f1   = f1_score(y_val, y_pred, average='weighted')
auc  = roc_auc_score(y_val, y_prob)
print(f"\nValidation  Accuracy={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
print(classification_report(y_val, y_pred, target_names=['Non-Anemic','Anemic']))

# ── 7. Save model ─────────────────────────────────────────────────────────────
model.save_model(SAVE_PATH)
print(f"Model saved -> {SAVE_PATH}")

# ── 8. Quick sanity check on preset values ────────────────────────────────────
presets = {
    'Normal': {'HCT':37.5,'HGB':12.8,'MCH':30.5,'MCHC':34.,'MCV':89.,'MPV':8.2,'PLT':260.,'RBC':4.6,'RDW':12.8,'WBC':7.8},
    'Mild Anemia': {'HCT':31.,'HGB':10.2,'MCH':26.5,'MCHC':31.5,'MCV':79.,'MPV':8.6,'PLT':220.,'RBC':3.8,'RDW':15.2,'WBC':8.5},
    'Moderate Anemia': {'HCT':27.,'HGB':8.7,'MCH':24.,'MCHC':29.5,'MCV':74.,'MPV':9.,'PLT':195.,'RBC':3.2,'RDW':17.1,'WBC':9.2},
    'Severe Anemia': {'HCT':24.,'HGB':6.8,'MCH':21.,'MCHC':27.5,'MCV':68.,'MPV':9.4,'PLT':170.,'RBC':2.7,'RDW':19.2,'WBC':10.4},
}
print("\n=== Sanity Check ===")
for name, vals in presets.items():
    arr = np.array([[vals[f] for f in FEATURES]], dtype=np.float32)
    arr_scaled = (arr - mean) / std
    prob = model.predict_proba(arr_scaled)[0, 1]
    label = "Anemic" if prob >= 0.5 else "Non-Anemic"
    print(f"  {name:<22s}: Anemic={prob:.1%}  =>  {label}")
