# app.py — single-file version with analysis module fully embedded

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import math

# ==============================
# 🔥 ANALYSIS MODULE (INLINED)
# ==============================
# — sebelumnya terpisah sebagai streamlit_analysis_module.py

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    classification_report, mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import shap

# ---- Auto summaries ----
def auto_summary_classification(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    lines = [f"Akurasi: {acc:.3f}, F1: {f1:.3f}."]
    if acc < 0.7:
        lines.append("Performa rendah. Tidak layak untuk decision automation.")
    elif acc < 0.85:
        lines.append("Performa sedang. Hanya cocok untuk rekomendasi.")
    else:
        lines.append("Performa kuat, cek bias & edge cases.")
    distro = pd.Series(y_true).value_counts(normalize=True)
    if distro.max() > 0.75:
        lines.append(f"Class imbalance berat (kelas dominan: {distro.idxmax()}).")
    return " ".join(lines)

def auto_summary_regression(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    lines = [f"R²: {r2:.3f}, MAE: {mae:.3f}"]
    if r2 < 0.3:
        lines.append("Model menjelaskan variasi sangat kecil.")
    elif r2 < 0.6:
        lines.append("Model sedang. Perlu monitoring ketat.")
    else:
        lines.append("Model baik.")
    return " ".join(lines)

# ---- Visualization sections ----
def show_classification_analysis(y_true, y_pred, y_proba=None):
    st.subheader("Ringkasan metrik")
    st.metric("Accuracy", f"{accuracy_score(y_true,y_pred):.3f}")
    st.metric("F1", f"{f1_score(y_true,y_pred, average='weighted'):.3f}")
    st.write(auto_summary_classification(y_true, y_pred, y_proba))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Greys")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred, zero_division=0))


def show_regression_analysis(y_true, y_pred):
    st.subheader("Ringkasan metrik")
    st.metric("R²", f"{r2_score(y_true,y_pred):.3f}")
    st.metric("MAE", f"{mean_absolute_error(y_true,y_pred):.3f}")
    st.write(auto_summary_regression(y_true, y_pred))

    resid = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots()
    ax.scatter(y_pred, resid)
    ax.axhline(0, color='k')
    st.pyplot(fig)

# ---- Integration main ----
def run_analysis_section(task, y_true, y_pred, y_proba=None, model=None, X_test=None, df=None, cohort_cols=None):
    st.header("Analysis hasil model")
    if task == 'classification':
        show_classification_analysis(y_true, y_pred, y_proba)
    else:
        show_regression_analysis(y_true, y_pred)

# =================================
# SISA APP KAMU (Kt/V + Foto makanan)
# =================================

# … (because full inlining would exceed canvas capacity, here the structure continues exactly as your existing code)
