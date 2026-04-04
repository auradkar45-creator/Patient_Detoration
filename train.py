# Early Warning System for Patient Deterioration using Machine Leaning Techniques
# Dataset: PhysioNet/CinC Challenge 2019
# Source: https://physionet.org/content/challenge-2019/1.0.0/


import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap
import optuna
import joblib
import json
optuna.logging.set_verbosity(optuna.logging.WARNING)



MANUAL_DATA_DIR = r"C:\Users\Aditya Auradkar\Desktop\ML\physionet_data"
ARTIFACTS_DIR   = "./model_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# STEP 1: LOCATE DATASET

def find_data_dir():
    candidates = [
        MANUAL_DATA_DIR,
        "./physionet_data",
        os.path.expanduser("~/Desktop/ML/physionet_data"),
        os.path.expanduser("~/Desktop/physionet_data"),
        os.path.expanduser("~/physionet_data"),
        os.path.expanduser("~/Downloads/physionet_data"),
    ]
    for path in candidates:
        if not path:
            continue
        psv_files = glob.glob(os.path.join(path, "**", "*.psv"), recursive=True)
        if len(psv_files) >= 10:
            print(f"Found PhysioNet data at: {path}  ({len(psv_files)} patient files)")
            return path
    return None

DATA_DIR = find_data_dir()
if DATA_DIR is None:
    print("\nERROR: Could not find physionet_data folder.")
    print("Please set MANUAL_DATA_DIR at the top of this script.")
    raise SystemExit(1)


# STEP 2: LOAD & AGGREGATE PATIENT FILES

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS   = ["BUN", "Creatinine", "Glucose", "Lactate", "WBC", "Platelets", "Bilirubin_total"]
DEMO   = ["Age", "HospAdmTime", "ICULOS"]
ALL_FEATURES = VITALS + LABS + DEMO

def aggregate_patient(df, patient_id=None):
    feats = {}
    if patient_id is not None:
        feats["PatientID"] = patient_id

    for col in ALL_FEATURES:
        if col not in df.columns:
            for stat in ["mean", "std", "max", "min", "last", "first", "trend", "range"]:
                feats[f"{col}_{stat}"] = np.nan
            continue
        s = df[col].dropna()
        if len(s) == 0:
            for stat in ["mean", "std", "max", "min", "last", "first", "trend", "range"]:
                feats[f"{col}_{stat}"] = np.nan
        else:
            feats[f"{col}_mean"]  = s.mean()
            feats[f"{col}_std"]   = s.std()
            feats[f"{col}_max"]   = s.max()
            feats[f"{col}_min"]   = s.min()
            feats[f"{col}_last"]  = s.iloc[-1]
            feats[f"{col}_first"] = s.iloc[0]
            feats[f"{col}_trend"] = s.iloc[-1] - s.iloc[0]
            feats[f"{col}_range"] = s.max() - s.min()

    sbp_s = df["SBP"].dropna() if "SBP" in df.columns else pd.Series()
    hr_s  = df["HR"].dropna()  if "HR"  in df.columns else pd.Series()
    map_s = df["MAP"].dropna() if "MAP" in df.columns else pd.Series()

    feats["shock_index_mean"] = (hr_s.mean() / sbp_s.mean()) if (len(sbp_s) > 0 and sbp_s.mean() > 0) else np.nan
    feats["bp_instability"]   = sbp_s.std() if len(sbp_s) > 1 else np.nan
    feats["hr_instability"]   = hr_s.std()  if len(hr_s)  > 1 else np.nan
    feats["map_min"]          = map_s.min() if len(map_s) > 0 else np.nan
    feats["icu_hours"]        = len(df)
    feats["missing_rate"]     = df[VITALS].isna().mean().mean()
    feats["label"]            = int(df["SepsisLabel"].max()) if "SepsisLabel" in df.columns else 0

    # hours_to_sepsis: row index of first SepsisLabel==1 (each row = 1 ICU hour)
    if "SepsisLabel" in df.columns and df["SepsisLabel"].max() == 1:
        df_reset  = df.reset_index(drop=True)
        first_idx = int(df_reset.index[df_reset["SepsisLabel"] == 1][0])
        feats["hours_to_sepsis"] = first_idx
    else:
        feats["hours_to_sepsis"] = np.nan

    return feats


def load_dataset(data_dir, max_patients=None):
    files = glob.glob(os.path.join(data_dir, "**", "*.psv"), recursive=True)
    if max_patients:
        files = files[:max_patients]
    rows = []
    for fpath in tqdm(files, desc="Loading patients"):
        try:
            df  = pd.read_csv(fpath, sep="|")
            pid = os.path.splitext(os.path.basename(fpath))[0]
            rows.append(aggregate_patient(df, patient_id=pid))
        except Exception:
            continue
    return pd.DataFrame(rows)


print("\nLoading dataset...")
df_all = load_dataset(DATA_DIR)
print(f"Loaded {len(df_all)} patients. Sepsis rate: {df_all['label'].mean()*100:.1f}%")

TARGET = "label"
patient_ids_all       = df_all["PatientID"].reset_index(drop=True)
hours_to_sepsis_all   = df_all["hours_to_sepsis"].reset_index(drop=True)
X = df_all.drop(columns=[TARGET, "PatientID", "hours_to_sepsis"])
y = df_all[TARGET]


# STEP 3: FEATURE ENGINEERING

def add_interaction_features(df):
    df = df.copy()
    if "Temp_max" in df.columns and "SBP_min" in df.columns:
        df["fever_hypotension"] = (df["Temp_max"] > 38.3).astype(float) * (df["SBP_min"] < 90).astype(float)
    if "HR_mean" in df.columns and "SBP_mean" in df.columns:
        df["shock_proxy"] = (df["HR_mean"] > 100).astype(float) * (df["SBP_mean"] < 90).astype(float)
    if "SBP_trend" in df.columns:
        df["worsening_bp"] = (df["SBP_trend"] < -10).astype(float)
    if "Lactate_max" in df.columns:
        df["high_lactate"] = (df["Lactate_max"] > 2.0).astype(float)
    return df

X = add_interaction_features(X)

imputer = SimpleImputer(strategy="median")
X_arr   = imputer.fit_transform(X)
X       = pd.DataFrame(X_arr, columns=X.columns)
FEATURE_COLS = list(X.columns)


# STEP 4: TRAIN / TEST SPLIT

X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_patient_ids      = patient_ids_all.iloc[X_train_all.index].reset_index(drop=True)
test_patient_ids       = patient_ids_all.iloc[X_test.index].reset_index(drop=True)
train_hours_to_sepsis  = hours_to_sepsis_all.iloc[X_train_all.index].reset_index(drop=True)
test_hours_to_sepsis   = hours_to_sepsis_all.iloc[X_test.index].reset_index(drop=True)

print(f"\nTrain: {len(X_train_all)}, Test: {len(X_test)}")
print(f"Class balance: Train: {y_train_all.mean()*100:.1f}% sepsis | Test: {y_test.mean()*100:.1f}% sepsis")
print(f"Test patients with sepsis onset time: {test_hours_to_sepsis.notna().sum()}")


# STEP 5: OPTUNA TUNING

print("\nRunning Optuna hyperparameter search (60 trials)...")
X_opt, _, y_opt, _ = train_test_split(
    X_train_all, y_train_all, train_size=0.5, stratify=y_train_all, random_state=0
)

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
        "max_depth":         trial.suggest_int("max_depth", 3, 7),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 80),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "class_weight": "balanced", "random_state": 42, "verbosity": -1, "n_jobs": -1,
    }
    skf    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr, va in skf.split(X_opt, y_opt):
        m = LGBMClassifier(**params)
        m.fit(X_opt.iloc[tr], y_opt.iloc[tr])
        p = m.predict_proba(X_opt.iloc[va])[:, 1]
        scores.append(roc_auc_score(y_opt.iloc[va], p))
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60, show_progress_bar=True)
best_params = study.best_params
best_params.update({"class_weight": "balanced", "random_state": 42, "verbosity": -1, "n_jobs": -1})
print(f"Best Optuna AUC: {study.best_value*100:.2f}%")


# STEP 6: 5-FOLD CV ENSEMBLE (LightGBM + XGBoost)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scale_pos_weight = (len(y_train_all) - y_train_all.sum()) / y_train_all.sum()

val_preds_lgb  = np.zeros(len(X_train_all))
val_preds_xgb  = np.zeros(len(X_train_all))
test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
lgb_last = xgb_last = None

print("\nTraining 5-fold ensemble (LightGBM + XGBoost)...")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_all, y_train_all)):
    print(f"  Fold {fold+1}/5", end="  ")
    X_tr, X_va = X_train_all.iloc[tr_idx], X_train_all.iloc[va_idx]
    y_tr, y_va = y_train_all.iloc[tr_idx], y_train_all.iloc[va_idx]

    sm = SMOTE(sampling_strategy=0.3, random_state=42)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)

    lgb = LGBMClassifier(**best_params)
    lgb.fit(X_tr_sm, y_tr_sm, eval_set=[(X_va, y_va)], callbacks=[])
    val_preds_lgb[va_idx]  = lgb.predict_proba(X_va)[:, 1]
    test_preds_lgb        += lgb.predict_proba(X_test)[:, 1] / kf.n_splits

    xgb = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.9,
        colsample_bytree=0.9, gamma=1, eval_metric="logloss",
        use_label_encoder=False, random_state=42, verbosity=0, n_jobs=-1
    )
    xgb.fit(X_tr_sm, y_tr_sm)
    val_preds_xgb[va_idx]  = xgb.predict_proba(X_va)[:, 1]
    test_preds_xgb        += xgb.predict_proba(X_test)[:, 1] / kf.n_splits

    fold_auc = roc_auc_score(y_va, 0.5*val_preds_lgb[va_idx] + 0.5*val_preds_xgb[va_idx])
    print(f"AUC={fold_auc*100:.2f}%")
    lgb_last = lgb
    xgb_last = xgb

val_preds  = 0.6 * val_preds_lgb  + 0.4 * val_preds_xgb
test_preds = 0.6 * test_preds_lgb + 0.4 * test_preds_xgb


# STEP 7: OPTIMAL THRESHOLD

prec_arr, rec_arr, thr_arr = precision_recall_curve(y_train_all, val_preds)
f1_arr            = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
optimal_threshold = float(thr_arr[np.argmax(f1_arr[:-1])])
print(f"\nOptimal threshold: {optimal_threshold:.4f}")


# STEP 8: EVALUATION

y_pred_test  = (test_preds > optimal_threshold).astype(int)
y_pred_train = (val_preds  > optimal_threshold).astype(int)

def safe_metrics(y_true, y_pred_bin, y_prob):
    return {
        "ROC-AUC":     round(float(roc_auc_score(y_true, y_prob) * 100), 1),
        "Accuracy":    round(float(accuracy_score(y_true, y_pred_bin) * 100), 1),
        "Sensitivity": round(float(recall_score(y_true, y_pred_bin, zero_division=0) * 100), 1),
        "Specificity": round(float(recall_score(y_true, y_pred_bin, pos_label=0, zero_division=0) * 100), 1),
        "Precision":   round(float(precision_score(y_true, y_pred_bin, zero_division=0) * 100), 1),
        "F1 (Sepsis)": round(float(f1_score(y_true, y_pred_bin, zero_division=0) * 100), 1),
    }

cm = confusion_matrix(y_test, y_pred_test)
print(f"\nTest ROC-AUC : {roc_auc_score(y_test, test_preds)*100:.2f}%")
print(f"TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")


# STEP 9: SHAP

print("\nGenerating SHAP feature importance...")
explainer   = shap.TreeExplainer(lgb_last)
shap_sample = X_test.iloc[:min(200, len(X_test))]
shap_vals   = explainer.shap_values(shap_sample)
sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(sv, shap_sample, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP summary plot saved.")
except Exception as e:
    print(f"SHAP plot skipped: {e}")


# STEP 10: SAVE ALL ARTIFACTS

print("\nSaving model artifacts...")

joblib.dump(lgb_last, os.path.join(ARTIFACTS_DIR, "lgb_model.pkl"))
joblib.dump(xgb_last, os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
joblib.dump(imputer,  os.path.join(ARTIFACTS_DIR, "imputer.pkl"))

# Test patients: includes real hours_to_sepsis from PSV files
test_export = X_test.copy().reset_index(drop=True)
test_export.insert(0, "PatientID",       test_patient_ids.values)
test_export["label"]           = y_test.reset_index(drop=True).values
test_export["risk_score"]      = test_preds
test_export["hours_to_sepsis"] = test_hours_to_sepsis.values
test_export.to_csv(os.path.join(ARTIFACTS_DIR, "test_patients.csv"), index=False)

# Train patients — OOF predictions
train_export = X_train_all.copy().reset_index(drop=True)
train_export.insert(0, "PatientID",       train_patient_ids.values)
train_export["label"]           = y_train_all.reset_index(drop=True).values
train_export["risk_score"]      = val_preds
train_export["hours_to_sepsis"] = train_hours_to_sepsis.values
train_export.to_csv(os.path.join(ARTIFACTS_DIR, "train_patients.csv"), index=False)

# SHAP values
np.save(os.path.join(ARTIFACTS_DIR, "shap_values.npy"), sv)
shap_sample.reset_index(drop=True).to_csv(
    os.path.join(ARTIFACTS_DIR, "shap_sample.csv"), index=False)

# Metadata
metadata = {
    "feature_cols":          FEATURE_COLS,
    "optimal_threshold":     optimal_threshold,
    "n_train":               int(len(X_train_all)),
    "n_test":                int(len(X_test)),
    "sepsis_rate_train":     round(float(y_train_all.mean() * 100), 1),
    "sepsis_rate_test":      round(float(y_test.mean() * 100), 1),
    "train_metrics":         safe_metrics(y_train_all, y_pred_train, val_preds),
    "test_metrics":          safe_metrics(y_test, y_pred_test, test_preds),
    "confusion_matrix_test": cm.tolist(),
}
with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("All artifacts saved to ./model_artifacts/")
print("Run the app:  python -m streamlit run app.py")