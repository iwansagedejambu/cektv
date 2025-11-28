import streamlit as st
from PIL import Image
import io
import hashlib
import numpy as np
import pandas as pd
import math
import typing

# ---------------------------
# Safe / lazy imports for heavy deps
# ---------------------------
HAS_SKLEARN = True
HAS_MATPLOTLIB = True
HAS_SHAP = True

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, average_precision_score,
        classification_report, mean_squared_error, mean_absolute_error, r2_score
    )
except Exception:
    HAS_SKLEARN = False
    # lightweight fallbacks to avoid NameError — actual logic checks HAS_SKLEARN before using
    def accuracy_score(*a, **k): return None
    def f1_score(*a, **k): return None
    def confusion_matrix(*a, **k): return None
    def classification_report(*a, **k): return "scikit-learn not installed"
    def mean_squared_error(*a, **k): return None
    def mean_absolute_error(*a, **k): return None
    def r2_score(*a, **k): return None

try:
    import matplotlib.pyplot as plt
except Exception:
    HAS_MATPLOTLIB = False

# shap is optional and heavy — try import but tolerate failure
try:
    import shap
except Exception:
    HAS_SHAP = False

# Helper: show warnings in UI (call early)
def show_missing_dependency_warnings():
    if not HAS_SKLEARN:
        st.warning("Analisis metrik dinonaktifkan: 'scikit-learn' tidak terpasang. Tambahkan scikit-learn ke requirements.txt untuk mengaktifkan.")
    if not HAS_MATPLOTLIB:
        st.warning("Plot dinonaktifkan: 'matplotlib' tidak terpasang.")
    if not HAS_SHAP:
        st.info("SHAP tidak tersedia (opsional). Interpretabilitas SHAP akan dinonaktifkan.")

# ==============================
# 🔥 ANALYSIS MODULE (INLINED)
# ==============================
# Auto summaries, visualizations, and run_analysis_section

def auto_summary_classification(y_true, y_pred, y_proba=None):
    if not HAS_SKLEARN:
        return "Analisis metrik tidak tersedia (scikit-learn tidak terpasang)."
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
    if y_proba is not None and HAS_SKLEARN:
        try:
            if getattr(y_proba, "ndim", 1) == 1 or (hasattr(y_proba, "shape") and y_proba.shape[1] == 2):
                probs = y_proba[:,1] if getattr(y_proba, "ndim", 1) > 1 else y_proba
                auc = roc_auc_score(y_true, probs)
                lines.append(f"ROC-AUC: {auc:.3f}.")
        except Exception:
            pass
    return " ".join(lines)

def auto_summary_regression(y_true, y_pred):
    if not HAS_SKLEARN:
        return "Analisis metrik regresi tidak tersedia (scikit-learn tidak terpasang)."
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

def show_classification_analysis(y_true, y_pred, y_proba=None):
    st.subheader("Ringkasan metrik")
    if HAS_SKLEARN:
        st.metric("Accuracy", f"{accuracy_score(y_true,y_pred):.3f}")
        st.metric("F1", f"{f1_score(y_true,y_pred, average='weighted'):.3f}")
    else:
        st.write("Metrik tidak tersedia (scikit-learn tidak terpasang).")
    st.write(auto_summary_classification(y_true, y_pred, y_proba))

    if HAS_SKLEARN and HAS_MATPLOTLIB:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Greys")
        # annotate
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred, zero_division=0))
    else:
        if not HAS_SKLEARN:
            st.info("Confusion matrix / classification report dinonaktifkan karena scikit-learn tidak terpasang.")
        elif not HAS_MATPLOTLIB:
            st.info("Confusion matrix dinonaktifkan karena matplotlib tidak terpasang.")

def show_regression_analysis(y_true, y_pred):
    st.subheader("Ringkasan metrik")
    if HAS_SKLEARN:
        st.metric("R²", f"{r2_score(y_true,y_pred):.3f}")
        st.metric("MAE", f"{mean_absolute_error(y_true,y_pred):.3f}")
    else:
        st.write("Metrik regresi tidak tersedia (scikit-learn tidak terpasang).")
    st.write(auto_summary_regression(y_true, y_pred))

    if HAS_MATPLOTLIB and HAS_SKLEARN:
        resid = np.array(y_true) - np.array(y_pred)
        fig, ax = plt.subplots()
        ax.scatter(y_pred, resid, alpha=0.5)
        ax.axhline(0, color='k')
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Residual")
        st.pyplot(fig)
    else:
        st.info("Plot residual dinonaktifkan karena matplotlib / scikit-learn tidak terpasang.")

def run_analysis_section(task, y_true, y_pred, y_proba=None, model=None, X_test=None, df=None, cohort_cols=None):
    """
    task: 'classification' or 'regression'
    y_true, y_pred: array-like
    y_proba: optional probabilities for classification
    model/X_test: optional for feature importance / SHAP
    df/cohort_cols: optional for cohort analysis
    """
    st.header("Analysis hasil model")
    if task == 'classification':
        show_classification_analysis(y_true, y_pred, y_proba)
    else:
        show_regression_analysis(y_true, y_pred)

    # Feature importance / SHAP (optional)
    if model is not None and X_test is not None:
        st.subheader("Feature analysis")
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            feats = list(X_test.columns) if hasattr(X_test, "columns") else [f"f{i}" for i in range(len(fi))]
            df_fi = pd.DataFrame({"feature": feats, "importance": fi}).sort_values("importance", ascending=False)
            st.dataframe(df_fi)
            if HAS_MATPLOTLIB:
                fig, ax = plt.subplots()
                df_fi.head(20).plot.barh(x="feature", y="importance", ax=ax, legend=False)
                st.pyplot(fig)
        else:
            st.write("Model tidak memiliki attribute feature_importances_. Coba SHAP atau permutation importance.")

        if st.checkbox("Tampilkan SHAP (mahal compute)") and HAS_SHAP and HAS_MATPLOTLIB:
            try:
                sample = X_test.sample(n=min(500, len(X_test)), random_state=42) if isinstance(X_test, pd.DataFrame) else X_test
                explainer = shap.Explainer(model, sample)
                shap_vals = explainer(sample)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                shap.summary_plot(shap_vals, sample, show=False)
                st.pyplot(bbox_inches='tight')
            except Exception as e:
                st.error(f"Error SHAP: {e}")
        elif st.checkbox("Tampilkan SHAP (mahal compute)"):
            st.info("SHAP atau plotting tidak tersedia pada environment ini.")

    # cohort analysis (requires df)
    if df is not None and cohort_cols:
        st.subheader("Cohort analysis")
        for c in cohort_cols:
            st.write(f"--- Cohort by {c} ---")
            try:
                grp = df.groupby(c).apply(lambda g: pd.Series({
                    'n': len(g),
                    'accuracy': accuracy_score(g['y_true'], g['y_pred']) if HAS_SKLEARN and len(g['y_true'].unique())>1 else np.nan,
                    'mae': mean_absolute_error(g['y_true'], g['y_pred']) if HAS_SKLEARN and g['y_true'].dtype.kind in 'fi' else np.nan
                })).reset_index()
                st.dataframe(grp.sort_values('n', ascending=False).head(50))
            except Exception as e:
                st.write('Error cohort analysis for', c, e)

# ============================
# SISA APP KAMU (Kt/V + Foto makanan)
# ============================
# Config / DB kecil nutrisi
FOOD_DB_PER_100G = {
    "rice":       {"kcal":130, "protein":2.7, "fat":0.3, "carb":28.0, "potassium":26,  "phosphate":43,  "calcium":10},
    "chicken":    {"kcal":239, "protein":27.0,"fat":14.0,"carb":0.0,  "potassium":256, "phosphate":200, "calcium":15},
    "egg":        {"kcal":155, "protein":13.0,"fat":11.0,"carb":1.1,  "potassium":126, "phosphate":198, "calcium":50},
    "tofu":       {"kcal":76,  "protein":8.0, "fat":4.8, "carb":1.9,  "potassium":121, "phosphate":136, "calcium":350},
    "banana":     {"kcal":89,  "protein":1.1, "fat":0.3, "carb":23.0, "potassium":358, "phosphate":22,  "calcium":5},
    "potato":     {"kcal":77,  "protein":2.0, "fat":0.1, "carb":17.0, "potassium":421, "phosphate":57,  "calcium":10},
    "salmon":     {"kcal":208, "protein":20.4,"fat":13.4,"carb":0.0,  "potassium":363, "phosphate":252, "calcium":9},
    "mixed_salad":{"kcal":20,  "protein":1.2, "fat":0.2, "carb":3.6,  "potassium":194, "phosphate":30,  "calcium":36},
    "sauce":      {"kcal":120, "protein":2.0, "fat":6.0, "carb":12.0, "potassium":50, "phosphate":30, "calcium":10},
    "sweet_potato":{"kcal":86, "protein":1.6, "fat":0.1, "carb":20.1,"potassium":337,"phosphate":47,"calcium":30},
    "unknown":    {"kcal":100, "protein":3.0, "fat":5.0, "carb":12.0, "potassium":50, "phosphate":40, "calcium":20},
}
FOOD_KEYS = list(FOOD_DB_PER_100G.keys())

# Prototypes
PROTOTYPES = {
    "rice":        {"rgb": (245,245,240), "tex": 0.03},
    "chicken":     {"rgb": (220,180,150), "tex": 0.08},
    "egg":         {"rgb": (250,230,140), "tex": 0.06},
    "tofu":        {"rgb": (245,245,235), "tex": 0.02},
    "banana":      {"rgb": (240,210,60),  "tex": 0.03},
    "potato":      {"rgb": (210,160,100), "tex": 0.05},
    "salmon":      {"rgb": (230,120,120), "tex": 0.09},
    "mixed_salad": {"rgb": (80,150,80),   "tex": 0.04},
    "sauce":       {"rgb": (90,45,30),    "tex": 0.01},
    "sweet_potato":{"rgb": (235,120,50),  "tex": 0.04},
    "unknown":     {"rgb": (128,128,128), "tex": 0.05},
}

# Utilities
def _rgb_dist(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i]))**2 for i in range(3)))

def _image_to_np(img: Image.Image, max_side=500):
    w,h = img.size
    if max(w,h) > max_side:
        scale = max_side / float(max(w,h))
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return np.array(img.convert("RGB"))

def _quantize_colors(img_np, k=4):
    pixels = img_np.reshape(-1, 3).astype(np.float32)
    # initialize centers using percentiles
    centers = []
    N = len(pixels)
    for p in np.linspace(0, 100, k+2)[1:-1]:
        idx = int(N * (p/100.0))
        centers.append(pixels[idx])
    centers = np.array(centers, dtype=np.float32)
    for _ in range(8):
        d = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        new_centers = []
        for i in range(k):
            sel = pixels[labels == i]
            if len(sel) == 0:
                new_centers.append(centers[i])
            else:
                new_centers.append(sel.mean(axis=0))
        new_centers = np.array(new_centers)
        if np.allclose(new_centers, centers, atol=1.0):
            break
        centers = new_centers
    label_map = labels.reshape(img_np.shape[0], img_np.shape[1])
    centers = centers.astype(np.uint8)
    return centers, label_map

def _local_texture_score(gray_np, mask):
    sel = gray_np[mask]
    if sel.size == 0:
        return 0.0
    return float(np.std(sel) / (np.mean(sel)+1e-6))

def _median_brightness(gray_np, mask):
    sel = gray_np[mask]
    if sel.size == 0: return 0.0
    return float(np.median(sel)/255.0)

# Cache analyze to avoid recompute for same bytes
@st.cache_data(show_spinner=False)
def analyze_food_image_bytes(image_bytes: bytes, n_clusters=4):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return analyze_food_image(img, n_clusters=n_clusters)

def analyze_food_image(image: Image.Image, n_clusters=4):
    img_np = _image_to_np(image, max_side=500)
    H,W,_ = img_np.shape
    gray = np.array(image.convert("L").resize((W,H)))  # align sizes

    centers, label_map = _quantize_colors(img_np, k=n_clusters)
    masks = []
    total_pixels = float(H*W)

    for i, c in enumerate(centers):
        mask = (label_map == i)
        area = float(mask.sum())
        if area < 50:
            continue
        avg_color = tuple(int(x) for x in c.tolist())
        tex = _local_texture_score(gray, mask)
        bright = _median_brightness(gray, mask)
        masks.append({"idx": i, "area": area, "area_frac": area/total_pixels, "color": avg_color, "tex": tex, "bright": bright})

    if not masks:
        return {"detected": [], "totals": {}, "img_shape": (H,W)}

    detected = []
    est_total_grams = 450.0  # heuristic plate mass
    for m in sorted(masks, key=lambda x: x["area"], reverse=True):
        best = None
        best_score = 1e9
        for name, proto in PROTOTYPES.items():
            color_score = _rgb_dist(m["color"], proto["rgb"])
            tex_score = abs(m["tex"] - proto["tex"]) * 200.0
            bright_score = abs(m["bright"] - 0.5) * 10.0 if name == "sauce" else 0.0
            score = color_score + tex_score + bright_score
            if score < best_score:
                best_score = score
                best = name
        portion_g = max(20, int(est_total_grams * m["area_frac"]))
        per100 = FOOD_DB_PER_100G.get(best, FOOD_DB_PER_100G["unknown"])
        factor = portion_g / 100.0
        detected.append({
            "label": best,
            "portion_g": portion_g,
            "kcal": round(per100["kcal"] * factor, 1),
            "protein_g": round(per100["protein"] * factor, 1),
            "fat_g": round(per100["fat"] * factor, 1),
            "carb_g": round(per100["carb"] * factor, 1),
            "potassium_mg": round(per100["potassium"] * factor, 1),
            "phosphate_mg": round(per100["phosphate"] * factor, 1),
            "calcium_mg": round(per100["calcium"] * factor, 1),
            "area_frac": m["area_frac"],
            "color": m["color"],
            "tex": m["tex"],
            "match_score": round(best_score, 2)
        })

    totals = {"kcal":0,"protein_g":0,"fat_g":0,"carb_g":0,"potassium_mg":0,"phosphate_mg":0,"calcium_mg":0}
    for d in detected:
        for k in totals:
            totals[k] += d[k]

    return {"detected": detected, "totals": totals, "img_shape": (H,W)}

# Kt/V calculator + Pernefri notes
def hitung_ktv(qb, durasi_jam, bb_kering):
    clearance = 0.7 * qb
    waktu_menit = durasi_jam * 60.0
    v = 0.55 * bb_kering * 1000.0
    if v <= 0:
        return 0.0
    ktv = (clearance * waktu_menit) / v
    return round(ktv, 2)

def pernefri_note(ktv):
    note = {"level": None, "message": [], "advice": []}
    if ktv >= 1.8:
        note["level"] = "Excellent"
        note["message"] = ["Kt/V sangat baik (≥1.8)."]
    elif 1.7 <= ktv < 1.8:
        note["level"] = "Adequate"
        note["message"] = ["Kt/V mencapai target (1.7–1.79)."]
    elif 1.4 <= ktv < 1.7:
        note["level"] = "Borderline"
        note["message"] = ["Kt/V borderline; perlu evaluasi lebih lanjut."]
        note["advice"] = [
            "Pertimbangkan menambah durasi dialisis 30–60 menit.",
            "Tinjau akses vaskular dan aliran (Qb).",
            "Evaluasi recirculation dan efisiensi dialyzer."
        ]
    else:
        note["level"] = "Inadequate"
        note["message"] = ["Kt/V di bawah target (<1.4)."]
        note["advice"] = [
            "Tambah waktu dialisis 30–90 menit tergantung toleransi.",
            "Tingkatkan Qb jika akses memungkinkan (target 250–300 mL/menit).",
            "Pertimbangkan penilaian ulang akses vaskular (fistula/graft).",
            "Diskusikan perubahan protokol dengan nephrologist."
        ]
    return note

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Kt/V + Food Photo (heuristic)", layout="wide")
st.title("Kt/V + Food Photo — Heuristic (no API)")
show_missing_dependency_warnings()

st.sidebar.header("Panduan singkat")
st.sidebar.write("""
- Analisis foto: heuristic segmen warna + tekstur. Koreksi manual disarankan.
- Untuk akurasi klinis, gunakan model terlatih atau layanan Vision API.
""")

left, right = st.columns([1,1])

# Left: Kt/V
with left:
    st.header("💉 Kalkulator Kt/V")
    qb = st.number_input("Laju Aliran Darah (Qb) — mL/menit", min_value=50, max_value=800, value=220, step=10)
    bb_kering = st.number_input("Berat Badan Kering (kg)", min_value=20.0, max_value=200.0, value=48.5, step=0.1)
    durasi_jam = st.number_input("Durasi Dialisis (jam)", min_value=0.5, max_value=12.0, value=4.0, step=0.25)
    if st.button("Hitung Kt/V"):
        ktv = hitung_ktv(qb, durasi_jam, bb_kering)
        st.subheader(f"Hasil Kt/V: **{ktv}**")
        note = pernefri_note(ktv)
        for line in note["message"]:
            st.info(line)
        if note["advice"]:
            st.markdown("**Rekomendasi:**")
            for a in note["advice"]:
                st.write("- " + a)

# Right: Food photo
with right:
    st.header("📸 Analisis Foto Makanan (Heuristic)")
    st.markdown("Unggah foto (jpg/png). Sistem akan menampilkan deteksi per-klaster, skor kecocokan, dan memungkinkan koreksi.")
    uploaded = st.file_uploader("Unggah foto makanan", type=["jpg","jpeg","png"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Preview", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
            image = None

        if image is not None:
            st.info("Menganalisis... (heuristic)")
            # use cached version that keys on bytes
            try:
                image_bytes = uploaded.getvalue()
                analysis = analyze_food_image_bytes(image_bytes, n_clusters=4)
            except Exception:
                analysis = analyze_food_image(image, n_clusters=4)

            detected = analysis.get("detected", [])
            totals = analysis.get("totals", {})

            if not detected:
                st.warning("Tidak terdeteksi item — coba foto lain (lebih terang/kontras).")
            else:
                st.markdown("### Deteksi (urut dari area terbesar)")
                edited = []
                for i,d in enumerate(detected):
                    st.markdown(f"**Item #{i+1}** — tebakan: **{d['label']}** (area {d['area_frac']:.2%})")
                    cols = st.columns([2,1,1,1,1])
                    # show small color swatch
                    cols[0].markdown(f"<div style='width:36px;height:18px;background:rgb{tuple(d['color'])};border:1px solid #444'></div>", unsafe_allow_html=True)
                    label_opt = FOOD_KEYS
                    sel_label = cols[0].selectbox(f"Label #{i+1}", options=label_opt, index=label_opt.index(d['label']) if d['label'] in label_opt else 0, key=f"label_{i}")
                    portion = cols[1].number_input(f"Porsi (g) #{i+1}", min_value=10, max_value=2000, value=int(d["portion_g"]), step=10, key=f"portion_{i}")
                    match = cols[2].number_input(f"Match score #{i+1}", min_value=0.0, value=float(d["match_score"]), step=0.01, key=f"score_{i}")
                    tex = cols[3].number_input(f"Tex #{i+1}", min_value=0.0, value=float(d["tex"]), step=0.001, key=f"tex_{i}")
                    proto_scores = []
                    for name, proto in PROTOTYPES.items():
                        sc = _rgb_dist(d["color"], proto["rgb"]) + abs(d["tex"] - proto["tex"])*200.0
                        proto_scores.append((name, sc))
                    proto_scores = sorted(proto_scores, key=lambda x: x[1])
                    alt = ", ".join([f"{p[0]}({p[1]:.1f})" for p in proto_scores[:3]])
                    cols[4].write("Alt: " + alt)

                    per100 = FOOD_DB_PER_100G.get(sel_label, FOOD_DB_PER_100G["unknown"])
                    factor = portion / 100.0
                    recomputed = {
                        "label": sel_label,
                        "portion_g": int(portion),
                        "kcal": round(per100["kcal"] * factor, 1),
                        "protein_g": round(per100["protein"] * factor, 1),
                        "fat_g": round(per100["fat"] * factor, 1),
                        "carb_g": round(per100["carb"] * factor, 1),
                        "potassium_mg": round(per100["potassium"] * factor, 1),
                        "phosphate_mg": round(per100["phosphate"] * factor, 1),
                        "calcium_mg": round(per100["calcium"] * factor, 1),
                    }
                    edited.append(recomputed)

                if st.button("Recalculate totals"):
                    df = pd.DataFrame(edited)
                    st.dataframe(df, width=700)
                    agg = {
                        "Energi (kcal)": df["kcal"].sum(),
                        "Protein (g)": df["protein_g"].sum(),
                        "Lemak (g)": df["fat_g"].sum(),
                        "Karbohidrat (g)": df["carb_g"].sum(),
                        "Kalium (mg)": df["potassium_mg"].sum(),
                        "Fosfat (mg)": df["phosphate_mg"].sum(),
                        "Kalsium (mg)": df["calcium_mg"].sum(),
                    }
                    st.markdown("**Total nutrisi (setelah koreksi):**")
                    st.write(agg)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download hasil (CSV)", data=csv, file_name="food_analysis_corrected.csv", mime="text/csv")
                else:
                    st.markdown("**Estimasi nutrisi (sementara):**")
                    st.write(totals)
                    df = pd.DataFrame([{
                        "Makanan": d["label"],
                        "Porsi (g)": d["portion_g"],
                        "Energi (kcal)": d["kcal"],
                        "Protein (g)": d["protein_g"],
                        "Lemak (g)": d["fat_g"],
                        "Karbohidrat (g)": d["carb_g"],
                        "Kalium (mg)": d["potassium_mg"],
                        "Fosfat (mg)": d["phosphate_mg"],
                        "Kalsium (mg)": d["calcium_mg"],
                    } for d in detected])
                    st.dataframe(df, width=700)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download hasil (CSV)", data=csv, file_name="food_analysis_raw.csv", mime="text/csv")
    else:
        st.info("Belum ada foto diunggah.")

st.markdown("---")
st.caption("⚠ Prototype: heuristic. Untuk akurasi tinggi butuh model ML atau Vision API. Gunakan koreksi manual untuk hasil klinis.")

# Developer: quick access to analysis module
with st.expander("Developer: Run analysis module (demo)"):
    st.write("Jika kamu punya model/prediksi, module analysis akan menampilkan metrik, confusion matrix, dan SHAP.")
    st.write("Di sini demo cepat menggunakan synthetic data agar kamu lihat layoutnya.")
    if st.button("Jalankan demo analysis (classification)"):
        if HAS_SKLEARN:
            y_true = np.random.choice([0,1,2], size=200, p=[0.5,0.3,0.2])
            y_pred = np.random.choice([0,1,2], size=200, p=[0.5,0.3,0.2])
            y_proba = np.random.rand(200,3)
            y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
            run_analysis_section('classification', y_true, y_pred, y_proba=y_proba, model=None, X_test=None, df=None, cohort_cols=None)
        else:
            st.info("Demo classification dinonaktifkan: scikit-learn tidak terpasang.")

    if st.button("Jalankan demo analysis (regression)"):
        if HAS_SKLEARN:
            y_true = np.random.randn(200) * 10 + 100
            y_pred = y_true + np.random.randn(200) * 8
            run_analysis_section('regression', y_true, y_pred, y_proba=None, model=None, X_test=None, df=None, cohort_cols=None)
        else:
            st.info("Demo regression dinonaktifkan: scikit-learn tidak terpasang.")
