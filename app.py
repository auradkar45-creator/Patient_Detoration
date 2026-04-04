import os
import json
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

ARTIFACTS_DIR = "./model_artifacts"

# Page Config
st.set_page_config(
    page_title="SepsisGuard AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }
    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #2563eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 0.5rem;
    }
    .metric-box.red    { border-left-color: #dc2626; }
    .metric-box.green  { border-left-color: #16a34a; }
    .metric-box.amber  { border-left-color: #f59e0b; }
    .metric-box.purple { border-left-color: #7c3aed; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:linear-gradient(135deg,#1e3a5f,#2563eb);
            padding:1.5rem 2rem;border-radius:14px;margin-bottom:1rem;'>
  <div style='font-size:2rem;font-weight:800;color:white;'>SepsisGuard AI</div>
  <div style='color:#93c5fd;font-size:0.95rem;margin-top:0.2rem;'>
    Early Warning System for Patient Deterioration &nbsp;&middot;&nbsp;
    PhysioNet/CinC 2019 &nbsp;&middot;&nbsp; XGBoost + LightGBM Ensemble
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### Configuration")

artifacts_available = os.path.exists(os.path.join(ARTIFACTS_DIR, "metadata.json"))

mode_options = ["Real Model (PhysioNet)", "Upload Patient Data"]
if not artifacts_available:
    mode_options = ["Upload Patient Data"]
    st.sidebar.warning("No trained model found. Run train_2.py first, or upload a CSV.")

mode      = st.sidebar.radio("Mode", mode_options)
threshold = st.sidebar.slider("Risk Threshold", 0.05, 0.95, 0.35, 0.01)
top_k_pct = st.sidebar.slider("Top-K Flag (%)", 1, 10, 2)
show_shap = st.sidebar.checkbox("Show SHAP Explanations", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("XGBoost + LightGBM soft-voting ensemble with SHAP explainability.")



# HELPERS

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "ROC-AUC":     round(roc_auc_score(y_true, y_prob) * 100, 1),
        "Accuracy":    round(accuracy_score(y_true, y_pred) * 100, 1),
        "Sensitivity": round(recall_score(y_true, y_pred, zero_division=0) * 100, 1),
        "Specificity": round(recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100, 1),
        "Precision":   round(precision_score(y_true, y_pred, zero_division=0) * 100, 1),
        "F1 (Sepsis)": round(f1_score(y_true, y_pred, zero_division=0) * 100, 1),
    }

def safe_get(row, name, default=0.0):
    """Get a column value, trying alternate names like _mean suffix."""
    for key in [name, name + "_mean"]:
        v = row.get(key, None) if hasattr(row, "get") else getattr(row, key, None)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
    return default

def train_simple_ensemble(X_tr, y_tr):
    pos_weight = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08,
        num_leaves=31, class_weight="balanced",
        subsample=0.85, colsample_bytree=0.85,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=pos_weight, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0, n_jobs=-1,
        eval_metric="logloss", use_label_encoder=False
    )
    lgbm.fit(X_tr, y_tr)
    xgb.fit(X_tr, y_tr)
    return lgbm, xgb

def soft_vote(lgbm, xgb, X):
    return (lgbm.predict_proba(X)[:, 1] + xgb.predict_proba(X)[:, 1]) / 2

def urgency_label(h):
    """Return a plain-text urgency label from hours value."""
    if pd.isna(h):
        return "Not available"
    h = float(h)
    if h <= 4:   return "CRITICAL  (< 4 h)"
    if h <= 8:   return "URGENT    (4 - 8 h)"
    if h <= 16:  return "MONITOR   (8 - 16 h)"
    return "STABLE    (> 16 h)"



# LOAD DATA BASED ON MODE

if mode == "Real Model (PhysioNet)":

    @st.cache_resource
    def load_models():
        lgb = joblib.load(os.path.join(ARTIFACTS_DIR, "lgb_model.pkl"))
        xgb = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
        return lgb, xgb

    @st.cache_data
    def load_patient_data():
        test_df  = pd.read_csv(os.path.join(ARTIFACTS_DIR, "test_patients.csv"))
        train_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "train_patients.csv"))
        with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
            meta = json.load(f)
        return test_df, train_df, meta

    with st.spinner("Loading model and patient data..."):
        lgbm_model, xgb_model = load_models()
        test_df, train_df, meta = load_patient_data()

    FEATURE_COLS = meta["feature_cols"]

    X_tr = train_df[FEATURE_COLS].fillna(0)
    y_tr = train_df["label"]
    X_te = test_df[FEATURE_COLS].fillna(0)
    y_te = test_df["label"]

    # Use saved blended risk scores (from train_2.py ensemble)
    train_probs = train_df["risk_score"].values
    test_probs  = test_df["risk_score"].values

    train_preds = (train_probs > threshold).astype(int)
    test_preds  = (test_probs  > threshold).astype(int)

    train_metrics = compute_metrics(y_tr, train_preds, train_probs)
    test_metrics  = compute_metrics(y_te, test_preds,  test_probs)

    # hours_to_sepsis is real data from PSV files
    if "hours_to_sepsis" not in test_df.columns:
        test_df["hours_to_sepsis"] = np.nan

else:
    # Upload mode
    uploaded = st.file_uploader("Upload patient CSV (must include 'label' column)", type="csv")
    if not uploaded:
        st.warning("Please upload a CSV file.")
        st.stop()

    df = pd.read_csv(uploaded)
    if "label" not in df.columns:
        st.error("CSV must contain a 'label' column (0 = no sepsis, 1 = sepsis).")
        st.stop()
    if "PatientID" not in df.columns:
        df.insert(0, "PatientID", [f"P{i+1:04d}" for i in range(len(df))])

    # Auto-detect feature columns (everything except meta columns)
    meta_cols    = {"PatientID", "label", "hours_to_sepsis", "risk_score"}
    FEATURE_COLS = [c for c in df.columns if c not in meta_cols]

    with st.spinner("Training ensemble on uploaded data..."):
        all_ids = df["PatientID"].values
        train_ids, test_ids = train_test_split(
            all_ids, test_size=0.2, random_state=42, stratify=df["label"].values
        )
        train_df = df[df["PatientID"].isin(train_ids)].reset_index(drop=True)
        test_df  = df[df["PatientID"].isin(test_ids)].reset_index(drop=True)

        med = train_df[FEATURE_COLS].median()
        X_tr = train_df[FEATURE_COLS].fillna(med)
        y_tr = train_df["label"]
        X_te = test_df[FEATURE_COLS].fillna(med)
        y_te = test_df["label"]

        lgbm_model, xgb_model = train_simple_ensemble(X_tr, y_tr)
        train_probs = soft_vote(lgbm_model, xgb_model, X_tr)
        test_probs  = soft_vote(lgbm_model, xgb_model, X_te)
        train_preds = (train_probs > threshold).astype(int)
        test_preds  = (test_probs  > threshold).astype(int)
        train_metrics = compute_metrics(y_tr, train_preds, train_probs)
        test_metrics  = compute_metrics(y_te, test_preds,  test_probs)

    if "hours_to_sepsis" not in test_df.columns:
        test_df["hours_to_sepsis"] = np.nan



# BUILD RESULT TABLE (shared across tabs)

def _get_col(df, *names, default=0.0):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([default] * len(df))

result_df = pd.DataFrame({
    "PatientID":       test_df["PatientID"].values,
    "Risk Score (%)":  (test_probs * 100).round(1),
    "Risk Level":      ["HIGH RISK" if p > threshold else "Low Risk" for p in test_probs],
    "True Label":      ["Sepsis"    if l == 1 else "No Sepsis" for l in y_te.values],
    "HR (bpm)":        _get_col(test_df, "HR_mean").round(1).values,
    "SBP (mmHg)":      _get_col(test_df, "SBP_mean").round(1).values,
    "Temp (C)":        _get_col(test_df, "Temp_max", "Temp_mean").round(1).values,
    "Shock Index":     _get_col(test_df, "shock_index", "shock_index_mean").round(2).values,
    "Resp Trend":      _get_col(test_df, "Resp_trend").round(2).values,
    "Hours to Sepsis": test_df["hours_to_sepsis"].values,
}).sort_values("Risk Score (%)", ascending=False).reset_index(drop=True)



# TABS

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Performance & Metrics",
    "Patient Risk Table",
    "SHAP Explainability",
    "Individual Patient",
    "Time-to-Sepsis",
])



# TAB 1 — Performance & Metrics

with tab1:
    st.markdown("### Model Performance — Train vs Test")

    metric_names = list(train_metrics.keys())
    col_m, col_roc = st.columns([1, 1])

    with col_m:
        st.markdown("#### Metrics Comparison Table")
        rows = [{
            "Metric":    m,
            "Train (%)": f"{train_metrics[m]:.1f}%",
            "Test (%)":  f"{test_metrics[m]:.1f}%",
            "Gap":       f"{train_metrics[m] - test_metrics[m]:+.1f}%",
        } for m in metric_names]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.markdown("#### Test Set — Metric Cards")
        card_styles = [("blue",""), ("green",""), ("red",""),
                       ("green",""), ("amber",""), ("purple","")]
        for m, (c, _) in zip(metric_names, card_styles):
            val = test_metrics[m]
            color_val = "#16a34a" if val >= 80 else "#f59e0b" if val >= 60 else "#dc2626"
            st.markdown(f"""
            <div class="metric-box {c}">
                <b>{m}</b>
                <span style='float:right;font-size:1.25rem;font-weight:700;color:{color_val};'>
                    {val:.1f}%
                </span>
            </div>""", unsafe_allow_html=True)

    with col_roc:
        st.markdown("#### ROC Curve — Train vs Test")
        fpr_tr, tpr_tr, _ = roc_curve(y_tr, train_probs)
        fpr_te, tpr_te, _ = roc_curve(y_te, test_probs)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr_tr, tpr_tr, color="#2563eb", lw=2, linestyle="--",
                label=f"Train AUC = {train_metrics['ROC-AUC']:.1f}%")
        ax.plot(fpr_te, tpr_te, color="#dc2626", lw=2.5,
                label=f"Test  AUC = {test_metrics['ROC-AUC']:.1f}%")
        ax.fill_between(fpr_te, tpr_te, alpha=0.09, color="#dc2626")
        ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="Random classifier")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
        ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Confusion Matrices — Train vs Test")
    cm1, cm2 = st.columns(2)

    def plot_cm(cm_mat, title, ax):
        labels = [["TN","FP"],["FN","TP"]]
        bg     = [["#dcfce7","#fee2e2"],["#fef9c3","#dbeafe"]]
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j,1-i),1,1,color=bg[i][j],zorder=0))
                ax.text(j+0.5, 1.5-i, f"{labels[i][j]}\n{cm_mat[i,j]}",
                        ha="center", va="center", fontsize=14,
                        fontweight="bold", color="#1e293b")
        ax.set_xlim(0,2); ax.set_ylim(0,2)
        ax.set_xticks([0.5,1.5])
        ax.set_xticklabels(["Predicted\nNegative","Predicted\nPositive"], fontsize=9)
        ax.set_yticks([0.5,1.5])
        ax.set_yticklabels(["Actual\nPositive","Actual\nNegative"], fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(length=0)

    with cm1:
        fig, ax = plt.subplots(figsize=(4,3.5))
        cm_train = confusion_matrix(y_tr, train_preds)
        plot_cm(cm_train, f"Training Set (n={len(y_tr)})", ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        tn,fp,fn,tp = cm_train.ravel()
        st.caption(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    with cm2:
        fig, ax = plt.subplots(figsize=(4,3.5))
        cm_test = confusion_matrix(y_te, test_preds)
        plot_cm(cm_test, f"Test Set (n={len(y_te)})", ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        tn,fp,fn,tp = cm_test.ravel()
        st.caption(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    st.markdown("---")
    st.markdown("#### All Metrics — Train vs Test Bar Chart")
    fig, ax = plt.subplots(figsize=(10,4))
    x = np.arange(len(metric_names)); w = 0.35
    b1 = ax.bar(x-w/2, [train_metrics[m] for m in metric_names], w,
                label="Train", color="#2563eb", alpha=0.85, zorder=3)
    b2 = ax.bar(x+w/2, [test_metrics[m]  for m in metric_names], w,
                label="Test",  color="#dc2626", alpha=0.85, zorder=3)
    ax.bar_label(b1, fmt="%.1f%%", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.1f%%", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 115); ax.set_ylabel("Score (%)")
    ax.set_title("Performance Metrics — Train vs Test", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.caption(
        f"Training set: {len(train_df)} patients  |  "
        f"Test set: {len(test_df)} patients  |  "
        f"Sepsis prevalence: train={y_tr.mean()*100:.1f}%  /  test={y_te.mean()*100:.1f}%"
    )



# TAB 2 — Patient Risk Table

with tab2:
    st.markdown("### Patient Risk Scores — Test Set")

    k = max(1, int(top_k_pct / 100 * len(result_df)))
    st.caption(f"{len(result_df)} test patients  |  Top {k} flagged  |  Threshold = {threshold:.2f}")

    filt = st.selectbox("Filter", ["All Patients", "HIGH RISK Only", "Sepsis Cases Only"])
    disp = result_df.copy()
    if filt == "HIGH RISK Only":      disp = disp[disp["Risk Level"] == "HIGH RISK"]
    elif filt == "Sepsis Cases Only": disp = disp[disp["True Label"] == "Sepsis"]

    def highlight_risk(row):
        if row["Risk Level"] == "HIGH RISK":
            return ["background-color:#fee2e2;font-weight:600"] * len(row)
        return [""] * len(row)

    st.dataframe(
        disp.style
            .apply(highlight_risk, axis=1)
            .format({
                "Risk Score (%)": "{:.1f}%",
                "Shock Index":    "{:.2f}",
                "Resp Trend":     "{:.2f}",
            }),
        use_container_width=True,
        height=400,
    )
    st.download_button(
        "Download as CSV",
        disp.to_csv(index=False).encode(),
        "sepsisguard_patients.csv",
        "text/csv",
    )



# TAB 3 — SHAP Explainability

with tab3:
    st.markdown("### SHAP Feature Importance")

    shap_png = os.path.join(ARTIFACTS_DIR, "shap_summary.png")
    if mode == "Real Model (PhysioNet)" and os.path.exists(shap_png):
        st.markdown("#### SHAP Summary (generated during training)")
        st.image(shap_png, use_container_width=True)
        st.caption("SHAP summary plot from train_2.py.  "
                   "Red = high feature value, Blue = low feature value.")
    elif not show_shap:
        st.info("Enable 'Show SHAP Explanations' in the sidebar.")
    else:
        with st.spinner("Computing SHAP values..."):
            try:
                explainer = shap.TreeExplainer(lgbm_model)
                shap_samp = X_te.iloc[:min(150, len(X_te))]
                sv_raw    = explainer.shap_values(shap_samp)
                sv        = sv_raw[1] if isinstance(sv_raw, list) else sv_raw

                mean_shap = (
                    pd.Series(np.abs(sv).mean(axis=0), index=FEATURE_COLS)
                    .sort_values(ascending=False)
                    .head(15)
                )

                col_bar, col_dot = st.columns(2)

                with col_bar:
                    st.markdown("#### Mean |SHAP| — Top 15 Features")
                    fig, ax = plt.subplots(figsize=(6,5))
                    pal  = ["#dc2626" if any(k in f for k in ["SBP","HR","shock","lactate","Resp"])
                            else "#2563eb" for f in mean_shap.index]
                    bars = ax.barh(mean_shap.index[::-1], mean_shap.values[::-1],
                                   color=pal[::-1], edgecolor="white")
                    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
                    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
                    ax.set_title("Top 15 Predictors", fontsize=11, fontweight="bold")
                    ax.spines[["top","right"]].set_visible(False)
                    ax.set_xlim(0, mean_shap.values.max() * 1.25)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                with col_dot:
                    st.markdown("#### SHAP Dot Plot")
                    top_feat = mean_shap.index.tolist()
                    sv_top   = sv[:, [FEATURE_COLS.index(f) for f in top_feat]]
                    fv_top   = shap_samp[top_feat].values
                    fig, ax  = plt.subplots(figsize=(6,5))
                    for i, feat in enumerate(top_feat[::-1]):
                        ci     = top_feat.index(feat)
                        sc_col = sv_top[:, ci]
                        fv_col = fv_top[:, ci]
                        fv_n   = (fv_col - fv_col.min()) / (np.ptp(fv_col) + 1e-9)
                        sc = ax.scatter(sc_col, np.ones(len(sc_col))*i,
                                        c=fv_n, cmap="RdBu_r", alpha=0.5, s=15, zorder=3)
                    ax.set_yticks(range(len(top_feat)))
                    ax.set_yticklabels(top_feat[::-1], fontsize=8)
                    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
                    ax.set_xlabel("SHAP value", fontsize=9)
                    ax.set_title("Feature Impact Distribution", fontsize=11, fontweight="bold")
                    ax.spines[["top","right"]].set_visible(False)
                    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
                    cbar.set_label("Feature value\n(low to high)", fontsize=8)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                st.markdown("""
---
#### Interpretation Guide
| Symbol | Meaning |
|--------|---------|
| Red bar  | Cardiovascular / shock feature |
| Blue bar | Respiratory / temperature feature |
| Positive SHAP | Feature raises sepsis risk |
| Negative SHAP | Feature lowers sepsis risk |
""")
            except Exception as e:
                st.warning(f"SHAP computation failed: {e}")



# TAB 4 — Individual Patient + Search

with tab4:
    st.markdown("### Individual Patient Risk Explainer")

    # Patient search
    st.markdown("**Patient ID Search**")
    patient_ids_list = result_df["PatientID"].tolist()

    search_col, browse_col = st.columns([1, 2])
    with search_col:
        search_query = st.text_input(
            "Search Patient ID",
            placeholder="e.g. p000123",
            label_visibility="collapsed",
        )

    filtered_ids = (
        [pid for pid in patient_ids_list if search_query.strip().lower() in str(pid).lower()]
        if search_query.strip() else patient_ids_list
    )
    if not filtered_ids:
        st.warning(f"No patient found matching '{search_query}'. Showing all patients.")
        filtered_ids = patient_ids_list

    with browse_col:
        selected = st.selectbox("Select Patient", filtered_ids, label_visibility="collapsed")

    # Quick search summary badge
    if search_query.strip():
        match_row = result_df[result_df["PatientID"] == selected]
        if len(match_row):
            mr = match_row.iloc[0]
            rc = "#dc2626" if mr["Risk Level"] == "HIGH RISK" else "#16a34a"
            st.markdown(
                f"<div style='padding:0.4rem 0.8rem;background:#f8fafc;"
                f"border-radius:8px;border-left:4px solid {rc};font-size:0.9rem;'>"
                f"Patient <b>{selected}</b> &nbsp;|&nbsp; Risk: "
                f"<span style='color:{rc};font-weight:700;'>{mr['Risk Score (%)']:.1f}%</span>"
                f" &nbsp;|&nbsp; {mr['Risk Level']} &nbsp;|&nbsp; True label: {mr['True Label']}"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Locate patient in test_df
    sel_mask = test_df["PatientID"] == selected
    if sel_mask.sum() == 0:
        st.error(f"Patient {selected} not found in the test set.")
        st.stop()

    sel_pos  = test_df.index[sel_mask][0]
    sel_prob = test_probs[sel_pos]
    sel_row  = test_df.iloc[sel_pos]

    risk_color = "#dc2626" if sel_prob > threshold else "#16a34a"
    risk_label = "HIGH RISK"   if sel_prob > threshold else "LOW RISK"
    risk_bg    = "#fee2e2"     if sel_prob > threshold else "#dcfce7"

    col_score, col_vitals, col_alerts = st.columns([1, 2, 1])

    with col_score:
        st.markdown(f"""
        <div style='text-align:center;padding:2rem;background:{risk_bg};
                    border:2.5px solid {risk_color};border-radius:14px;'>
            <div style='font-size:3rem;font-weight:800;color:{risk_color};'>
                {sel_prob*100:.1f}%
            </div>
            <div style='font-size:1.1rem;font-weight:700;color:{risk_color};'>{risk_label}</div>
            <div style='color:#64748b;font-size:0.82rem;margin-top:0.4rem;'>Sepsis Risk Score</div>
            <hr style='border-color:{risk_color};opacity:0.3;'>
            <div style='font-size:0.85rem;color:#475569;'>
                True label: <b>{"Sepsis" if sel_row.get("label",0)==1 else "No Sepsis"}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_vitals:
        st.markdown("**Key Vitals**")
        v1, v2, v3 = st.columns(3)
        v1.metric("Heart Rate",    f"{safe_get(sel_row,'HR_mean'):.0f} bpm",
                  delta=f"{safe_get(sel_row,'HR_trend'):+.1f} trend")
        v2.metric("Systolic BP",   f"{safe_get(sel_row,'SBP_mean'):.0f} mmHg",
                  delta=f"{safe_get(sel_row,'SBP_trend'):+.1f} trend", delta_color="inverse")
        v3.metric("Temperature",   f"{safe_get(sel_row,'Temp_max'):.1f} C")
        v4, v5, v6 = st.columns(3)
        v4.metric("O2 Saturation", f"{safe_get(sel_row,'O2Sat_mean',98):.1f}%",
                  delta=f"{safe_get(sel_row,'O2Sat_trend'):+.1f}", delta_color="inverse")
        v5.metric("Resp Rate",     f"{safe_get(sel_row,'Resp_mean'):.0f} /min",
                  delta=f"{safe_get(sel_row,'Resp_trend'):+.1f} trend")
        si = safe_get(sel_row,"shock_index") or safe_get(sel_row,"shock_index_mean")
        v6.metric("Shock Index",   f"{si:.2f}",
                  delta="Elevated" if si > 1.0 else "Normal",
                  delta_color="inverse" if si > 1.0 else "normal")

    with col_alerts:
        st.markdown("**Clinical Alerts**")
        si    = safe_get(sel_row, "shock_index") or safe_get(sel_row, "shock_index_mean")
        flags = []
        if si > 1.0:                                  flags.append(("WARN",  "Elevated Shock Index",  "#f59e0b"))
        if safe_get(sel_row,"Temp_max") > 38.3:       flags.append(("WARN",  "Fever (> 38.3 C)",      "#f59e0b"))
        if safe_get(sel_row,"SBP_mean",120) < 90:     flags.append(("ALERT", "Hypotension (< 90 mmHg)","#dc2626"))
        if safe_get(sel_row,"high_lactate"):           flags.append(("ALERT", "Elevated Lactate",      "#dc2626"))
        if safe_get(sel_row,"HR_mean") > 100:         flags.append(("WARN",  "Tachycardia (> 100 bpm)","#f59e0b"))
        if safe_get(sel_row,"O2Sat_mean",100) < 94:   flags.append(("ALERT", "Low O2 Sat (< 94%)",    "#dc2626"))
        if safe_get(sel_row,"Resp_trend") > 1.5:      flags.append(("WARN",  "Rising Resp Rate",       "#f59e0b"))
        if flags:
            for tag, msg, col in flags:
                st.markdown(
                    f"<div style='background:#fff;border-left:3px solid {col};"
                    f"padding:0.3rem 0.6rem;border-radius:6px;margin-bottom:0.3rem;"
                    f"font-size:0.84rem;'>"
                    f"<b>[{tag}]</b> {msg}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("No critical alerts.")

    # All feature values (expandable)
    with st.expander("View All Feature Values for This Patient"):
        feat_display = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Value":   [round(float(sel_row.get(f, np.nan)), 4) if pd.notna(sel_row.get(f, np.nan)) else "N/A"
                        for f in FEATURE_COLS],
        })
        st.dataframe(feat_display, hide_index=True, height=320, use_container_width=True)

    # SHAP waterfall
    if show_shap:
        st.markdown("---")
        st.markdown("#### SHAP Waterfall — Feature Contributions for This Patient")
        with st.spinner("Computing SHAP..."):
            try:
                exp_ind = shap.TreeExplainer(lgbm_model)
                te_idx  = list(test_df.index).index(sel_pos)
                sv_ind  = exp_ind.shap_values(X_te.iloc[[te_idx]])
                sv_one  = sv_ind[1][0] if isinstance(sv_ind, list) else sv_ind[0]
                contrib = (
                    pd.Series(sv_one, index=FEATURE_COLS)
                    .sort_values(key=abs, ascending=False)
                    .head(12)
                )
                fig, ax = plt.subplots(figsize=(9, 4.5))
                bar_colors = ["#dc2626" if v > 0 else "#2563eb" for v in contrib.values[::-1]]
                bars = ax.barh(contrib.index[::-1], contrib.values[::-1],
                               color=bar_colors, edgecolor="white")
                ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
                ax.axvline(0, color="black", lw=1)
                ax.set_xlabel("SHAP contribution", fontsize=10)
                ax.set_title(
                    f"Individual SHAP  |  Patient: {selected}  |  Risk: {sel_prob*100:.1f}%",
                    fontsize=11, fontweight="bold"
                )
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.caption("Red = pushes risk HIGHER     Blue = pushes risk LOWER")
            except Exception as e:
                st.warning(f"SHAP computation failed: {e}")



# TAB 5 — Time-to-Sepsis

with tab5:
    st.markdown("### Time-to-Sepsis Analysis and Triage")
    st.caption(
        "Hours to sepsis onset = ICU hour at which SepsisLabel first became 1 in the patient record. "
        "Non-sepsis patients do not have an onset time and are excluded from this view."
    )

    high_df = result_df[result_df["Risk Level"] == "HIGH RISK"].copy()

    if len(high_df) == 0:
        st.warning("No HIGH RISK patients at the current threshold. Lower the Risk Threshold in the sidebar.")
    else:
        hrs_map = dict(zip(test_df["PatientID"], test_df["hours_to_sepsis"]))
        high_df["Sepsis Onset Hour"] = high_df["PatientID"].map(hrs_map)
        high_df["Urgency"]           = high_df["Sepsis Onset Hour"].apply(urgency_label)

        # Separate patients with and without onset time
        has_onset = high_df[high_df["Sepsis Onset Hour"].notna()].copy()
        no_onset  = high_df[high_df["Sepsis Onset Hour"].isna()].copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total HIGH RISK",     len(high_df))
        c2.metric("With Sepsis Onset",   len(has_onset))
        c3.metric("No Sepsis Onset",     len(no_onset),
                  help="Patients flagged HIGH RISK but did not develop sepsis in this record.")
        c4.metric("Critical (< 4 h)",    has_onset["Urgency"].str.startswith("CRITICAL").sum())

        st.markdown("---")

        if len(has_onset) > 0:
            has_onset = has_onset.sort_values("Sepsis Onset Hour")

            col_t, col_h = st.columns([1.5, 1])

            with col_t:
                st.markdown("#### Sepsis Triage Queue (patients with confirmed onset)")
                tdf = has_onset[[
                    "PatientID", "Risk Score (%)", "Urgency",
                    "Sepsis Onset Hour", "HR (bpm)", "SBP (mmHg)"
                ]].copy()
                tdf["Sepsis Onset Hour"] = tdf["Sepsis Onset Hour"].apply(lambda x: f"{x:.0f} h")

                def color_urg(row):
                    t = row["Urgency"]
                    if t.startswith("CRITICAL"): return ["background-color:#fee2e2"] * len(row)
                    if t.startswith("URGENT"):   return ["background-color:#ffedd5"] * len(row)
                    if t.startswith("MONITOR"):  return ["background-color:#fef9c3"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    tdf.style.apply(color_urg, axis=1),
                    use_container_width=True,
                    height=320,
                )
                st.download_button(
                    "Download Triage CSV",
                    tdf.to_csv(index=False).encode(),
                    "sepsisguard_triage.csv",
                    "text/csv",
                )

            with col_h:
                st.markdown("#### Sepsis Onset Hour Distribution")
                valid = has_onset["Sepsis Onset Hour"].astype(float)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(valid, bins=15, color="#dc2626", alpha=0.8, edgecolor="white")
                ax.axvline(4,  color="#b91c1c", lw=1.5, linestyle="--", label="Critical (4 h)")
                ax.axvline(8,  color="#f97316", lw=1.5, linestyle="--", label="Urgent (8 h)")
                ax.axvline(16, color="#eab308", lw=1.5, linestyle="--", label="Monitor (16 h)")
                ax.set_xlabel("ICU Hour of Sepsis Onset", fontsize=10)
                ax.set_ylabel("Patients", fontsize=10)
                ax.set_title("Intervention Window", fontsize=11, fontweight="bold")
                ax.legend(fontsize=8)
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        if len(no_onset) > 0:
            with st.expander(f"View {len(no_onset)} HIGH RISK patients without confirmed sepsis onset"):
                st.caption("These patients were flagged HIGH RISK by the model "
                           "but did not have a SepsisLabel=1 in their record. "
                           "They may be false positives or pre-sepsis cases requiring monitoring.")
                st.dataframe(
                    no_onset[["PatientID","Risk Score (%)","True Label","HR (bpm)","SBP (mmHg)"]],
                    use_container_width=True,
                )


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.8rem;padding:0.5rem;'>
    SepsisGuard AI &nbsp;&middot;&nbsp; ML Hackathon 2026 &nbsp;&middot;&nbsp;
    Early Warning System for Patient Deterioration &nbsp;&middot;&nbsp;
    Dataset: <a href='https://physionet.org/content/challenge-2019/1.0.0/'
    target='_blank' style='color:#60a5fa;'>PhysioNet/CinC 2019</a>
    &nbsp;&middot;&nbsp; Ensemble: XGBoost + LightGBM &nbsp;&middot;&nbsp; Explainability: SHAP
</div>
""", unsafe_allow_html=True)