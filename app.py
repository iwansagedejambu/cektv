import streamlit as st
import io
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime
from huggingface_hub import InferenceClient

# ============ CONFIG HALAMAN ============
st.set_page_config(page_title="Kt/V & Food Tracker (AI)", layout="wide")

# ====== LOAD API KEY HUGGING FACE DARI STREAMLIT SECRETS ======
try:
    HF_TOKEN = st.secrets["default"]["HF_TOKEN"]
    client = InferenceClient(token=HF_TOKEN)
except Exception:
    st.error("HF API Key belum diset di secrets!")
    st.stop()

# ============ KALKULATOR Kt/V ============
def hitung_ktv(qb, durasi_jam, bb_kering):
    clearance = 0.7 * qb
    waktu_menit = durasi_jam * 60
    v = 0.55 * bb_kering * 1000
    if v <= 0:
        return 0
    return round((clearance * waktu_menit) / v, 2)

def pernefri_note(ktv):
    if ktv >= 1.8:
        return "Sangat baik (≥1.8)", []
    elif ktv >= 1.7:
        return "Adekuat (1.7–1.79)", []
    elif ktv >= 1.4:
        return "Borderline (1.4–1.69)", [
            "Tambahkan durasi 30–60 menit",
            "Periksa akses vaskular / aliran Qb",
            "Evaluasi efisiensi dialyzer"
        ]
    else:
        return "Tidak adekuat (<1.4)", [
            "Tambah durasi 30–90 menit",
            "Tingkatkan Qb bila memungkinkan",
            "Periksa kembali akses vaskular"
        ]

# ============ USDA TRACKER SEMENTARA (DUMMY) ============
@st.cache_data
def load_usda_simple():
    return pd.DataFrame()

USDA_DF = load_usda_simple()

def get_nutrition_from_usda_by_desc(desc: str, grams: float):
    if USDA_DF.empty:
        return None
    df = USDA_DF[USDA_DF["description"].str.contains(desc, case=False, na=False)]
    if df.empty:
        return None
    row = df.iloc[0]
    factor = grams / 100.0
    return {
        "description": row["description"],
        "kcal": round(row["kcal"] * factor, 1),
        "protein": round(row["protein"] * factor, 1),
        "fat": round(row["fat"] * factor, 1),
        "carb": round(row["carb"] * factor, 1),
        "k": round(row["k"] * factor, 1),
        "p": round(row["p"] * factor, 1),
        "ca": round(row["ca"] * factor, 1),
    }

# ============ AI FOOD PREDICTION LEWAT API HF ============
def ai_food_predict(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    try:
        result = client.post(
            json={
                "inputs": buf.getvalue()
            },
            model="nateraw/food"  # Model food recognition ringan
        )
        if "label" in result and result["label"]:
            return result["label"]
        return "unknown"
    except Exception:
        return "unknown"

# ============ FOOD LOG ============
if "food_log" not in st.session_state:
    st.session_state.food_log = []

def add_food(label, grams):
    nut = get_nutrition_from_usda_by_desc(label, grams)
    if nut is None:
        # fallback default
        nut = {
            "description": label,
            "kcal": round(100*(grams/100),1),
            "protein": round(3*(grams/100),1),
            "fat": round(5*(grams/100),1),
            "carb": round(12*(grams/100),1),
            "k": round(50*(grams/100),1),
            "p": round(40*(grams/100),1),
            "ca": round(20*(grams/100),1),
        }

    entry = {
        "time": datetime.now().strftime("%H:%M"),
        "food": nut["description"],
        "grams": grams,
        "kcal": nut["kcal"],
        "protein": nut["protein"],
        "fat": nut["fat"],
        "carb": nut["carb"],
        "k": nut["k"],
        "p": nut["p"],
        "ca": nut["ca"],
    }
    st.session_state.food_log.append(entry)

# =====================================
# UI HALAMAN (2 KOLOM)
# =====================================
c1, c2 = st.columns([1,1])

# ---- KALKULATOR Kt/V ----
with c1:
    st.header("💉 Kalkulator Kt/V")
    qb = st.number_input("Qb (mL/menit)", 250, 10)
    dur = st.number_input("Durasi (jam)", 4.0, 0.25)
    bb  = st.number_input("BB kering (kg)", 50.0, 1.0)

    if st.button("Hitung Kt/V", key="btn_ktv"):
        ktv = hitung_ktv(qb, dur, bb)
        status, adv = pernefri_note(ktv)
        st.subheader(f"Kt/V = **{ktv}**")
        st.write(f"Status: **{status}**")
        if adv:
            st.warning("Rekomendasi:")
            for a in adv:
                st.write("•", a)

# ---- UPLOAD & ANALISIS FOTO MAKANAN ----
with c2:
    st.header("📸 Analisis Foto Makanan (AI)")
    up = st.file_uploader("Upload foto makanan", ["jpg","png","jpeg"])

    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Foto makanan")

        label = ai_food_predict(img)
        st.success(f"Terdeteksi AI: **{label}**")

        # User override supaya lebih logis untuk makanan campur
        final_label = st.selectbox(
            "Jika makanan campur, kamu bisa ganti label di sini:",
            ["unknown", "Nasi", "Ayam", "Rendang", "Sayur", label],
            index=0,
            key="override_food"
        )

        grams = st.number_input("Perkiraan Berat (gram)", 150, 10, key="grams_food")

        if st.button("Tambah ke log makanan dari foto", key="btn_add_food"):
            add_food(final_label, grams)
            st.success("✅ Berhasil ditambahkan ke log makanan!")

# ---- LOG MAKANAN ----
st.markdown("---")
st.header("📒 Log Asupan Harian")

if st.session_state.food_log:
    df = pd.DataFrame(st.session_state.food_log)
    st.dataframe(df)

    totals = df[["kcal","protein","fat","carb","k","p","ca"]].sum()
    st.subheader("📊 Total nutrisi hari ini (perkiraan)")
    st.json({
        "Energi (kcal)" : float(totals["kcal"]),
        "Protein (g)"   : float(totals["protein"]),
        "Lemak (g)"     : float(totals["fat"]),
        "Karbo (g)"     : float(totals["carb"]),
        "Kalium (mg)"   : float(totals["k"]),
        "Fosfor (mg)"   : float(totals["p"]),
        "Kalsium (mg)"  : float(totals["ca"]),
    })
else:
    st.info("Belum ada makanan tercatat.")
