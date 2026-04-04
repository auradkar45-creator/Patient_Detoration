import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ================================
# 1. LOAD DATA
# ================================
train_df = pd.read_csv("train_patients.csv")
test_df = pd.read_csv("test_patients.csv")
shap_df = pd.read_csv("shap_sample.csv")

print("\n✅ Files loaded successfully")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("SHAP shape:", shap_df.shape)

# ================================
# 2. AUTO DETECT COLUMNS
# ================================
def find_col(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None

y_true_col = find_col(test_df, ["true", "label", "target"])
y_pred_col = find_col(test_df, ["risk", "prob", "score"])

if y_true_col is None or y_pred_col is None:
    raise ValueError("❌ Could not detect required columns. Check column names.")

print(f"\nDetected columns:")
print(f"True Label → {y_true_col}")
print(f"Prediction Score → {y_pred_col}")

y_true = test_df[y_true_col]
y_pred_prob = test_df[y_pred_col]

# ================================
# 3. THRESHOLD
# ================================
threshold = 0.35
y_pred = (y_pred_prob >= threshold).astype(int)

# ================================
# 4. METRICS
# ================================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_prob)

print("\n📊 MODEL PERFORMANCE")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# ================================
# 5. CONFUSION MATRIX
# ================================
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

TN, FP, FN, TP = cm.ravel()

print("\n📌 Confusion Matrix Values:")
print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

# ================================
# 6. ROC CURVE
# ================================
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc_val = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

# ================================
# 7. PRECISION-RECALL CURVE
# ================================
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_prob)

plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("pr_curve.png")
plt.close()

# ================================
# 8. METRICS BAR CHART
# ================================
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC-AUC": roc_auc
}

plt.figure()
plt.bar(metrics.keys(), metrics.values())
plt.title("Model Performance Metrics")
plt.savefig("metrics_bar.png")
plt.close()

# ================================
# 9. RISK DISTRIBUTION
# ================================
plt.figure()
plt.hist(y_pred_prob, bins=50)
plt.xlabel("Risk Score")
plt.ylabel("Frequency")
plt.title("Risk Score Distribution")
plt.savefig("risk_distribution.png")
plt.close()

# ================================
# 10. SHAP FEATURE IMPORTANCE
# ================================
# Remove non-feature columns
exclude_cols = ["PatientID", y_true_col, y_pred_col]

shap_features = shap_df.drop(columns=[c for c in shap_df.columns if c in exclude_cols], errors='ignore')

# Mean absolute SHAP
shap_importance = shap_features.abs().mean().sort_values(ascending=False)

top_n = 15
top_features = shap_importance.head(top_n)

plt.figure()
top_features.plot(kind="bar")
plt.title("Top SHAP Feature Importance")
plt.savefig("shap_importance.png")
plt.close()

print("\n🔍 Top SHAP Features:")
print(top_features)

# ================================
# 11. TRAIN vs TEST COMPARISON
# ================================
train_y_true = train_df[y_true_col]
train_y_pred_prob = train_df[y_pred_col]
train_y_pred = (train_y_pred_prob >= threshold).astype(int)

train_metrics = {
    "Accuracy": accuracy_score(train_y_true, train_y_pred),
    "Precision": precision_score(train_y_true, train_y_pred),
    "Recall": recall_score(train_y_true, train_y_pred),
    "F1": f1_score(train_y_true, train_y_pred),
}

test_metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
}

labels = list(train_metrics.keys())

train_vals = list(train_metrics.values())
test_vals = list(test_metrics.values())

x = np.arange(len(labels))

plt.figure()
plt.bar(x - 0.2, train_vals, width=0.4, label='Train')
plt.bar(x + 0.2, test_vals, width=0.4, label='Test')

plt.xticks(x, labels)
plt.title("Train vs Test Metrics")
plt.legend()
plt.savefig("train_vs_test.png")
plt.close()

print("\n✅ All plots generated successfully!")