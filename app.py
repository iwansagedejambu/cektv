# app.py
import streamlit as st
from PIL import Image
import io
import hashlib
import pandas as pd
import numpy as np
import cv2

# ---------------------------
# Nutrition DB (per 100g)
# ---------------------------
FOOD_DB_PER_100G = {
    "rice":      {"kcal": 130, "protein": 2.7, "fat": 0.3,  "carb": 28.0, "potassium": 26,  "phosphate": 43,  "calcium": 10},
    "chicken":   {"kcal": 239, "protein": 27.0,"fat": 14.0, "carb": 0.0,  "potassium": 256, "phosphate": 200, "calcium": 15},
    "egg":       {"kcal": 155, "protein": 13.0,"fat": 11.0, "carb": 1.1,  "potassium": 126, "phosphate": 198, "calcium": 50},
    "tofu":      {"kcal": 76,  "protein": 8.0, "fat": 4.8,  "carb": 1.9,  "potassium": 121, "phosphate": 136, "calcium": 350},
    "banana":    {"kcal": 89,  "protein": 1.1, "fat": 0.3,  "carb": 23.0, "potassium": 358, "phosphate": 22,  "calcium": 5},
    "potato":    {"kcal": 77,  "protein": 2.0, "fat": 0.1,  "carb": 17.0, "potassium": 421, "phosphate": 57,  "calcium": 10},
    "salmon":    {"kcal": 208, "protein": 20.4,"fat": 13.4, "carb": 0.0,  "potassium": 363, "phosphate": 252, "calcium": 9},
    "mixed_salad":{"kcal":20,  "protein": 1.2, "fat": 0.2,  "carb": 3.6,  "potassium": 194, "phosphate": 30,  "calcium": 36},
    # add more if needed
}
# allow some extra labels for manual correction
EXTRA_LABELS = ["sweet_potato", "sauce", "unknown"]
FOOD_KEYS = list(FOOD_DB_PER_100G.keys()) + EXTRA_LABELS

# ---------------------------
# CV-based analyzer (color + edge heuristics)
# ---------------------------
def _rgb_to_vec(rgb):
    return np.array(rgb, dtype=np.float32)

def _color_distance(c1, c2):
    return np.linalg.norm(_rgb_to_vec(c1) - _rgb_to_vec(c2))

# prototypes: approximate RGB (R,G,B) and expected edge density
FOOD_PROTOTYPES = {
    "rice":        {"color": (245,245,240), "edge": 0.06, "per100": FOOD_DB_PER_100G["rice"]},
    "chicken":     {"color": (220,180,150), "edge": 0.10, "per100": FOOD_DB_PER_100G["chicken"]},
    "egg":         {"color": (250,230,140), "edge": 0.08, "per100": FOOD_DB_PER_100G["egg"]},
    "tofu":        {"color": (245,245,235), "edge": 0.03, "per100": FOOD_DB_PER_100G["tofu"]},
    "banana":      {"color": (240,210,60),  "edge": 0.04, "per100": FOOD_DB_PER_100G["banana"]},
    "potato":      {"color": (210,160,100), "edge": 0.07, "per100": FOOD_DB_PER_100G["potato"]},
    "salmon":      {"color": (230,120,120), "edge": 0.12, "per100": FOOD_DB_PER_100G["salmon"]},
    "mixed_salad": {"color": (80,150,80),   "edge": 0.05, "per100": FOOD_DB_PER_100G["mixed_salad"]},
    "sauce":       {"color": (90,45,30),    "edge": 0.02, "per100": {"kcal":120,"protein":2,"fat":6,"carb":12,"potassium":50,"phosphate":30,"calcium":10}},
    "sweet_potato":{"color": (235,120,50),  "edge": 0.06, "per100": {"kcal":86,"protein":1.6,"fat":0.1,"carb":20.1,"potassium":337,"phosphate":47,"calcium":30}},
    "unknown":     {"color": (128,128,128), "edge": 0.06, "per100": {"kcal":100,"protein":3,"fat":5,"carb":12,"potassium":50,"phosphate":40,"calcium":20}},
}

def analyze_food_image_cv_only(image: Image.Image, n_clusters=4):
    """
    Heuristic analyzer:
    - segments image colors with k-means
    - computes edge density per cluster
    - matches clusters to FOOD_PROTOTYPES by color+edge
    - returns list of detected items and aggregated totals
    """
    # PIL -> numpy RGB
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    # resize for speed (maintain aspect)
    scale = max(1, int(max(h, w) / 400))
    small = cv2.resize(arr, (w//scale, h//scale), interpolation=cv2.INTER_AREA)
    small_rgb = small.copy()

    # blur mildly
    blurred = cv2.GaussianBlur(small_rgb, (5,5), 0)
    data = blurred.reshape((-1,3)).astype(np.float32)

    n_clusters = max(2, min(6, n_clusters))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(data, n_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    centers = centers.astype(np.uint8)

    masks = []
    H, W = blurred.shape[:2]
    labels_img = labels.reshape(H, W)

    gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    for i in range(n_clusters):
        mask = (labels_img == i).astype(np.uint8)
        area = int(mask.sum())
        if area == 0:
            continue
        edge_density = float((edges * mask).sum()) / (area + 1e-9)
        avg_color = centers[i][:]
        # avg_color is (R,G,B)
        masks.append({"mask": mask, "area": area, "edge": edge_density, "color": (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))})

    if not masks:
        return {"detected": [], "totals": {}, "img_shape": (h,w)}

    masks.sort(key=lambda x: x["area"], reverse=True)
    total_area = float(sum([m["area"] for m in masks])) + 1e-9

    detected = []
    est_total_grams = 450.0  # heuristic total food on plate

    for m in masks:
        # find best prototype
        best = (None, 1e9)
        for pname, proto in FOOD_PROTOTYPES.items():
            dcol = _color_distance(m["color"], proto["color"])
            dedge = abs(m["edge"] - proto["edge"]) * 200.0
            score = dcol + dedge
            if score < best[1]:
                best = (pname, score)
        label = best[0] or "unknown"
        frac = m["area"] / total_area
        portion_g = max(20, int(est_total_grams * frac))
        per100 = FOOD_PROTOTYPES[label]["per100"]
        factor = portion_g / 100.0
        detected.append({
            "label": label,
            "portion_g": portion_g,
            "kcal": round(per100["kcal"] * factor, 1),
            "protein_g": round(per100["protein"] * factor, 1),
            "fat_g": round(per100["fat"] * factor, 1),
            "carb_g": round(per100["carb"] * factor, 1),
            "potassium_mg": round(per100["potassium"] * factor, 1),
            "phosphate_mg": round(per100["phosphate"] * factor, 1),
            "calcium_mg": round(per100["calcium"] * factor, 1),
            "area_frac": frac
        })

    totals = {"kcal":0,"protein_g":0,"fat_g":0,"carb_g":0,"potassium_mg":0,"phosphate_mg":0,"calcium_mg":0}
    for d in detected:
        for k in totals:
            totals[k] += d[k]

    detected.sort(key=lambda x: x["area_frac"], reverse=True)
    return {"detected": detected, "totals": totals, "img_shape": (h,w)}

# ---------------------------
# Kt/V calculator
# ---------------------------
def hitung_ktv(qb, durasi_jam, bb_kering):
    clearance = 0.7 * qb  # mL/min (assumption)
    waktu_menit = durasi_jam * 60.0
    v = 0.55 * bb_kering * 1000.0  # mL
    if v <= 0:
        return 0.0
    ktv = (clearance * waktu_menit) / v
    return round(ktv, 2)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Kt/V + Food Nutrition (prototype)", layout="wide")
st.title("Hemodialysis Kt/V + Photo Nutrition (improved prototype)")

st.sidebar.header("Panduan singkat")
st.sidebar.write("""
- Kalkulator Kt/V: masukkan Qb, BB kering, durasi. Ini estimasi kasar.  
- Analisis foto makanan: heuristic CV + koreksi manual. Bukan analisis klinis.  
- Jika terdeteksi salah, ubah label/porsi di tabel lalu tekan Recalculate.
""")

col1, col2 = st.columns([1, 1])

# LEFT: Kt/V
with col1:
    st.header("💉 Kalkulator Kt/V")
    qb = st.number_input("Laju Aliran Darah (Qb) — mL/menit", min_value=50, max_value=800, value=220, step=10)
    bb_kering = st.number_input("Berat Badan Kering (kg)", min_value=20.0, max_value=200.0, value=48.5, step=0.1)
    durasi_jam = st.number_input("Durasi Dialisis (jam)", min_value=0.5, max_value=12.0, value=4.0, step=0.25)
    if st.button("Hitung Kt/V"):
        ktv = hitung_ktv(qb, durasi_jam, bb_kering)
        st.subheader(f"Hasil perhitungan Kt/V: **{ktv}**")
        if ktv >= 1.7:
            st.success("✅ Kt/V tercapai (≥1.7).")
        elif 1.4 <= ktv < 1.7:
            st.warning("⚠ Kt/V mendekati target (1.4–1.7). Evaluasi diperlukan.")
        else:
            st.error("❌ Kt/V di bawah target (<1.4).")

    with st.expander("Penjelasan rumus dan asumsi"):
        st.markdown("""
        **Rumus sederhana:** Kt/V = (Clearance × Waktu) / V  
        Asumsi:
        - Clearance = 0.7 × Qb (mL/menit)
        - V = 0.55 × BB (kg) × 1000 (mL)
        **Disclaimer:** Perhitungan kasar. Bukan pengganti keputusan klinis.
        """)

# RIGHT: Food analyzer
with col2:
    st.header("📸 Analisis Gizi dari Foto Makanan — Improved")
    st.markdown("Unggah foto (jpg/png). Sistem akan mendeteksi beberapa komponen & hitung nutrisi. Koreksi manual disarankan.")

    uploaded = st.file_uploader("Unggah foto makanan", type=["jpg","jpeg","png"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Preview", use_column_width=True)
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
            image = None

        if image is not None:
            with st.spinner("Menganalisis (CV heuristic)..."):
                analysis = analyze_food_image_cv_only(image, n_clusters=4)

            detected = analysis.get("detected", [])
            totals = analysis.get("totals", {})
            st.subheader("Deteksi item (koreksi manual tersedia)")
            if not detected:
                st.info("Tidak terdeteksi item (gambar mungkin buram atau gelap). Coba foto lain.")
            else:
                # allow user to edit each detected item
                edited = []
                for i, d in enumerate(detected):
                    st.markdown(f"**Item #{i+1} (area fraction {d['area_frac']:.2f})**")
                    cols = st.columns([2,1,1,1])
                    # label selectbox
                    label_options = FOOD_KEYS
                    sel_label = cols[0].selectbox(f"Label #{i+1}", options=label_options, index=label_options.index(d["label"]) if d["label"] in label_options else len(label_options)-1, key=f"label_{i}")
                    # portion input
                    portion = cols[1].number_input(f"Porsi (g) #{i+1}", min_value=10, max_value=2000, value=int(d["portion_g"]), step=10, key=f"portion_{i}")
                    # show kcal
                    kcal = cols[2].number_input(f"Energi (kcal) #{i+1}", min_value=0.0, value=float(d["kcal"]), step=0.1, key=f"kcal_{i}")
                    # show potassium (read-only style)
                    pot = cols[3].number_input(f"K (mg) #{i+1}", min_value=0.0, value=float(d["potassium_mg"]), step=1.0, key=f"pot_{i}")

                    # if user changed label, recompute nutrients from DB
                    if sel_label in FOOD_DB_PER_100G:
                        per100 = FOOD_DB_PER_100G[sel_label]
                    else:
                        # extra labels mapping
                        per100 = FOOD_PROTOTYPES.get(sel_label, FOOD_PROTOTYPES["unknown"])["per100"]

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
                    # aggregate edited list
                    agg = {"Energi (kcal)":0,"Protein (g)":0,"Lemak (g)":0,"Karbohidrat (g)":0,"Kalium (mg)":0,"Fosfat (mg)":0,"Kalsium (mg)":0}
                    rows = []
                    for e in edited:
                        rows.append({
                            "Makanan": e["label"],
                            "Porsi (g)": e["portion_g"],
                            "Energi (kcal)": e["kcal"],
                            "Protein (g)": e["protein_g"],
                            "Lemak (g)": e["fat_g"],
                            "Karbohidrat (g)": e["carb_g"],
                            "Kalium (mg)": e["potassium_mg"],
                            "Fosfat (mg)": e["phosphate_mg"],
                            "Kalsium (mg)": e["calcium_mg"],
                        })
                        agg["Energi (kcal)"] += e["kcal"]
                        agg["Protein (g)"] += e["protein_g"]
                        agg["Lemak (g)"] += e["fat_g"]
                        agg["Karbohidrat (g)"] += e["carb_g"]
                        agg["Kalium (mg)"] += e["potassium_mg"]
                        agg["Fosfat (mg)"] += e["phosphate_mg"]
                        agg["Kalsium (mg)"] += e["calcium_mg"]

                    df = pd.DataFrame(rows)
                    st.dataframe(df, width=700)
                    st.markdown("**Total nutrisi (setelah koreksi):**")
                    st.write(agg)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download hasil (CSV)", data=csv, file_name="food_analysis_corrected.csv", mime="text/csv")

                else:
                    # show default aggregated totals from heuristic
                    st.markdown("**Estimasi nutrisi (sementara, sebelum koreksi):**")
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
st.caption("⚠ Prototype: ini peningkatan heuristic. Untuk akurasi tinggi butuh model ML + estimasi porsi yang tervalidasi.")
