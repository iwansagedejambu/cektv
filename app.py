# app.py
import streamlit as st
from PIL import Image
import io
import hashlib
import pandas as pd
import numpy as np

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
}
EXTRA_LABELS = ["sweet_potato", "sauce", "unknown"]
FOOD_KEYS = list(FOOD_DB_PER_100G.keys()) + EXTRA_LABELS

# ---------------------------
# Heuristic prototypes (RGB colors & edge expectation)
# ---------------------------
def _rgb_to_vec(rgb):
    return np.array(rgb, dtype=np.float32)

def _color_distance(c1, c2):
    return np.linalg.norm(_rgb_to_vec(c1) - _rgb_to_vec(c2))

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

# ---------------------------
# Pure-Pillow + NumPy analyzer (no cv2)
# ---------------------------
from PIL import ImageFilter

def _sobel_edge_density(pil_img_gray):
    """Return normalized edge density (0..1) using simple sobel kernels on grayscale numpy array."""
    arr = np.array(pil_img_gray, dtype=np.float32)
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    pad = np.pad(arr, ((1,1),(1,1)), mode='edge')
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    H, W = arr.shape
    for i in range(H):
        for j in range(W):
            region = pad[i:i+3, j:j+3]
            gx[i,j] = np.sum(region * Kx)
            gy[i,j] = np.sum(region * Ky)
    grad = np.hypot(gx, gy)
    thr = np.percentile(grad, 70)
    edges = (grad > thr).astype(np.float32)
    return float(edges.mean())

def analyze_food_image_cv_only(image: Image.Image, n_clusters=4):
    """
    Pure-Pillow + NumPy heuristic:
    - quantize colors with PIL adaptive palette
    - compute per-cluster area, avg color, and edge density proxy
    - match clusters to FOOD_PROTOTYPES by color+edge heuristics
    """
    w, h = image.size
    max_side = 400
    if max(w,h) > max_side:
        scale = max_side / float(max(w,h))
        new_size = (int(w*scale), int(h*scale))
        img_small = image.resize(new_size, Image.LANCZOS)
    else:
        img_small = image.copy()
    img_rgb = img_small.convert("RGB")

    # quantize to n_clusters colors using adaptive palette
    pal = img_rgb.convert("P", palette=Image.ADAPTIVE, colors=n_clusters).convert("RGB")
    pal_arr = np.array(pal)
    H, W = pal_arr.shape[:2]
    orig = np.array(img_rgb)
    # collect palette colors from pal image
    palette_colors = []
    for y in range(H):
        for x in range(W):
            c = tuple(int(v) for v in pal_arr[y,x])
            if c not in palette_colors:
                palette_colors.append(c)
    while len(palette_colors) < n_clusters:
        palette_colors.append((128,128,128))

    # assign pixels to nearest palette color (vectorized)
    flat_orig = orig.reshape(-1,3).astype(np.float32)
    palette_arr = np.array(palette_colors, dtype=np.float32)
    dists = np.linalg.norm(flat_orig[:, None, :] - palette_arr[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1).reshape(H, W)

    gray = img_small.convert("L")
    edge_density_full = _sobel_edge_density(gray)

    masks = []
    for k in range(len(palette_colors)):
        mask = (labels == k).astype(np.uint8)
        area = int(mask.sum())
        if area == 0:
            continue
        avg_color = palette_colors[k]
        # approximate local edge density using overall edge as proxy
        local_edge = edge_density_full
        masks.append({"mask": mask, "area": area, "edge": local_edge, "color": avg_color})

    if not masks:
        return {"detected": [], "totals": {}, "img_shape": (h,w)}

    masks.sort(key=lambda x: x["area"], reverse=True)
    total_area = float(sum([m["area"] for m in masks])) + 1e-9
    detected = []
    est_total_grams = 450.0

    for m in masks:
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
    clearance = 0.7 * qb
    waktu_menit = durasi_jam * 60.0
    v = 0.55 * bb_kering * 1000.0
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

with col2:
    st.header("📸 Analisis Gizi dari Foto Makanan — Improved")
    st.markdown("Unggah foto (jpg/png). Sistem akan mendeteksi beberapa komponen & hitung nutrisi. Koreksi manual disarankan.")
    uploaded = st.file_uploader("Unggah foto makanan", type=["jpg","jpeg","png"], accept_multiple_files=False)

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Preview", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
            image = None

        if image is not None:
            with st.spinner("Menganalisis (heuristic)..."):
                analysis = analyze_food_image_cv_only(image, n_clusters=4)

            detected = analysis.get("detected", [])
            totals = analysis.get("totals", {})
            st.subheader("Deteksi item (koreksi manual tersedia)")
            if not detected:
                st.info("Tidak terdeteksi item (gambar mungkin buram atau gelap). Coba foto lain.")
            else:
                edited = []
                for i, d in enumerate(detected):
                    st.markdown(f"**Item #{i+1} (area fraction {d['area_frac']:.2f})**")
                    cols = st.columns([2,1,1,1])
                    label_options = FOOD_KEYS
                    sel_label = cols[0].selectbox(f"Label #{i+1}", options=label_options, index=label_options.index(d["label"]) if d["label"] in label_options else len(label_options)-1, key=f"label_{i}")
                    portion = cols[1].number_input(f"Porsi (g) #{i+1}", min_value=10, max_value=2000, value=int(d["portion_g"]), step=10, key=f"portion_{i}")
                    kcal = cols[2].number_input(f"Energi (kcal) #{i+1}", min_value=0.0, value=float(d["kcal"]), step=0.1, key=f"kcal_{i}")
                    pot = cols[3].number_input(f"K (mg) #{i+1}", min_value=0.0, value=float(d["potassium_mg"]), step=1.0, key=f"pot_{i}")

                    if sel_label in FOOD_DB_PER_100G:
                        per100 = FOOD_DB_PER_100G[sel_label]
                    else:
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
st.caption("⚠ Prototype: heuristic. Untuk akurasi tinggi butuh model ML + estimasi porsi tervalidasi.")
