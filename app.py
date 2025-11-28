# app.py
import streamlit as st
from PIL import Image
import io
import hashlib
import pandas as pd

# ---------------------------
# Deterministic mock food analyzer (placeholder)
# ---------------------------
FOOD_DB_PER_100G = {
    "rice":      {"kcal": 130, "protein": 2.7, "fat": 0.3, "carb": 28.0, "potassium": 26, "phosphate": 43, "calcium": 10},
    "chicken":   {"kcal": 239, "protein": 27.0, "fat": 14.0, "carb": 0.0, "potassium": 256, "phosphate": 200, "calcium": 15},
    "egg":       {"kcal": 155, "protein": 13.0, "fat": 11.0, "carb": 1.1, "potassium": 126, "phosphate": 198, "calcium": 50},
    "tofu":      {"kcal": 76,  "protein": 8.0,  "fat": 4.8, "carb": 1.9, "potassium": 121, "phosphate": 136, "calcium": 350},
    "banana":    {"kcal": 89,  "protein": 1.1,  "fat": 0.3, "carb": 23.0, "potassium": 358, "phosphate": 22, "calcium": 5},
    "potato":    {"kcal": 77,  "protein": 2.0, "fat": 0.1, "carb": 17.0, "potassium": 421, "phosphate": 57, "calcium": 10},
    "salmon":    {"kcal": 208, "protein": 20.4, "fat": 13.4, "carb": 0.0, "potassium": 363, "phosphate": 252, "calcium": 9},
    "mixed_salad":{"kcal": 20, "protein": 1.2,  "fat": 0.2, "carb": 3.6, "potassium": 194, "phosphate": 30, "calcium": 36},
}
FOOD_KEYS = list(FOOD_DB_PER_100G.keys())

def _pick_food_label_from_bytes(img_bytes: bytes) -> str:
    h = hashlib.sha256(img_bytes).hexdigest()
    idx = int(h[:8], 16) % len(FOOD_KEYS)
    return FOOD_KEYS[idx]

def analyze_food_image(image: Image.Image):
    """
    Deterministic mock analyzer:
    - chooses a food label based on hash of image bytes (so same image -> same label)
    - assumes a default portion in grams
    - computes nutrition from FOOD_DB_PER_100G scaled by portion
    Replace with real model/API for production use.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    label = _pick_food_label_from_bytes(img_bytes)

    # default portion guess (g)
    default_portion = 150 if label in ("rice", "chicken", "salmon", "potato") else 100
    per100 = FOOD_DB_PER_100G[label]
    factor = default_portion / 100.0
    result = {
        "label": label,
        "portion_g": default_portion,
        "kcal": round(per100["kcal"] * factor, 1),
        "protein_g": round(per100["protein"] * factor, 1),
        "fat_g": round(per100["fat"] * factor, 1),
        "carb_g": round(per100["carb"] * factor, 1),
        "potassium_mg": round(per100["potassium"] * factor, 1),
        "phosphate_mg": round(per100["phosphate"] * factor, 1),
        "calcium_mg": round(per100["calcium"] * factor, 1),
    }
    return result

# ---------------------------
# Kt/V calculator
# ---------------------------
def hitung_ktv(qb, durasi_jam, bb_kering):
    # basic formula based on your earlier function
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
st.title("Hemodialysis Kt/V + Food Photo Nutrition (prototype)")

st.sidebar.header("Panduan singkat")
st.sidebar.write("""
- Kalkulator Kt/V: masukkan Qb, BB kering, durasi. Ini estimasi kasar.  
- Analisis foto makanan: prototype mock untuk demo UI saja — bukan analisis klinis.  
Jangan masukkan data pasien yang dapat diidentifikasi.
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
            st.warning("⚠ Kt/V mendekati target (1.4–1.7). Evaluasi lebih lanjut diperlukan.")
        else:
            st.error("❌ Kt/V di bawah target (<1.4). Pertimbangkan penyesuaian.")
    with st.expander("Penjelasan rumus dan asumsi"):
        st.markdown("""
        **Rumus sederhana:** Kt/V = (Clearance × Waktu) / V  
        Asumsi:
        - Clearance = 0.7 × Qb (mL/menit) — asumsi kasar
        - V (volume distribusi) = 0.55 × BB (kg) × 1000 (mL)
        **Disclaimer:** Perhitungan kasar. Bukan pengganti keputusan klinis.
        """)

with col2:
    st.header("📸 Analisis Gizi dari Foto Makanan — Prototype")
    st.markdown("Unggah foto makanan (jpg/png). Sistem akan menampilkan label perkiraan dan tabel nutrisi (mock).")
    uploaded = st.file_uploader("Unggah foto makanan", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Preview", use_column_width=True)
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
            image = None

        if image is not None:
            if st.button("Analisis Gizi (mock)"):
                with st.spinner("Menganalisis..."):
                    res = analyze_food_image(image)
                st.markdown(f"**Prediksi makanan (mock):** `{res['label']}` — perkiraan porsi **{res['portion_g']} g**")
                new_portion = st.slider("Atur porsi (gram) untuk kalkulasi ulang:", 20, 800, int(res["portion_g"]))
                # if user changes portion, res is scaled
                if new_portion != res["portion_g"]:
                    scale = new_portion / res["portion_g"]
                    for k in ("kcal","protein_g","fat_g","carb_g","potassium_mg","phosphate_mg","calcium_mg"):
                        res[k] = round(res[k] * scale, 1)
                    res["portion_g"] = new_portion

                df = pd.DataFrame([{
                    "Makanan": res["label"],
                    "Porsi (g)": res["portion_g"],
                    "Energi (kcal)": res["kcal"],
                    "Protein (g)": res["protein_g"],
                    "Lemak (g)": res["fat_g"],
                    "Karbohidrat (g)": res["carb_g"],
                    "Kalium (mg)": res["potassium_mg"],
                    "Fosfat (mg)": res["phosphate_mg"],
                    "Kalsium (mg)": res["calcium_mg"],
                }])
                st.dataframe(df, width=700)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download hasil (CSV)", data=csv, file_name="food_analysis.csv", mime="text/csv")
    else:
        st.info("Belum ada foto diunggah.")

st.markdown("---")
st.caption("⚠ Prototype — analisis gizi berbasis foto tidak akurat tanpa model & estimasi porsi. Gunakan hanya untuk demo.")
