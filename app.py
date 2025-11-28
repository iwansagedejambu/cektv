import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import io
import math
import json
from datetime import datetime

st.set_page_config(page_title="Kt/V & Food Tracker", layout="wide")

# =====================================
# 1) KALKULATOR Kt/V
# =====================================

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

# =====================================
# 2) NUTRISI DATABASE (LOCAL)
# =====================================
# "Per 100g" makanan umum
NUTRI_DB = {
    "rice": {"kcal":130,"protein":2.7,"fat":0.3,"carb":28,"k":26,"p":43,"ca":10},
    "chicken": {"kcal":239,"protein":27,"fat":14,"carb":0,"k":256,"p":200,"ca":15},
    "egg": {"kcal":155,"protein":13,"fat":11,"carb":1.1,"k":126,"p":198,"ca":50},
    "tofu": {"kcal":76,"protein":8,"fat":4.8,"carb":1.9,"k":121,"p":136,"ca":350},
    "banana": {"kcal":89,"protein":1.1,"fat":0.3,"carb":23,"k":358,"p":22,"ca":5},
    "salmon": {"kcal":208,"protein":20.4,"fat":13,"carb":0,"k":363,"p":252,"ca":9},
    "mixed_salad": {"kcal":20,"protein":1.2,"fat":0.2,"carb":3.6,"k":194,"p":30,"ca":36},
    "unknown": {"kcal":100,"protein":3,"fat":5,"carb":12,"k":50,"p":40,"ca":20},
}

PROTOTYPES = {
    "rice": (245,245,240),
    "chicken": (220,180,150),
    "egg": (250,230,140),
    "tofu": (245,245,235),
    "banana": (240,210,60),
    "salmon": (230,120,120),
    "mixed_salad": (80,150,80),
}

def rgb_dist(a,b):
    return sum((a[i]-b[i])**2 for i in range(3))**0.5

def analyze_food(image: Image.Image):
    img = image.resize((200,200))
    arr = np.array(img)

    pixels = arr.reshape(-1,3)
    mean_color = pixels.mean(axis=0)

    best = None
    best_score = 99999
    for name, proto in PROTOTYPES.items():
        d = rgb_dist(mean_color, proto)
        if d < best_score:
            best_score = d
            best = name

    return best or "unknown"

# =====================================
# 3) FOOD LOG (SESSION)
# =====================================
if "food_log" not in st.session_state:
    st.session_state.food_log = []

def add_food(label, grams):
    nut = NUTRI_DB.get(label, NUTRI_DB["unknown"])
    factor = grams / 100
    entry = {
        "time": datetime.now().strftime("%H:%M"),
        "food": label,
        "grams": grams,
        "kcal": round(nut["kcal"] * factor,1),
        "protein": round(nut["protein"] * factor,1),
        "fat": round(nut["fat"] * factor,1),
        "carb": round(nut["carb"] * factor,1),
        "k": round(nut["k"] * factor,1),
        "p": round(nut["p"] * factor,1),
        "ca": round(nut["ca"] * factor,1),
    }
    st.session_state.food_log.append(entry)

# =====================================
# UI — TIGA KOLOM UTAMA
# =====================================
col1, col2 = st.columns([1,1])

# -------------------------------------
# KIRI — KALKULATOR Kt/V
# -------------------------------------
with col1:
    st.header("💉 Kalkulator Kt/V Hemodialisis")

    qb = st.number_input("Aliran darah (Qb, mL/menit)", value=250, step=10)
    dur = st.number_input("Durasi (jam)", value=4.0, step=0.25)
    bb = st.number_input("Berat badan kering (kg)", value=50.0)

    if st.button("Hitung Kt/V"):
        ktv = hitung_ktv(qb, dur, bb)
        status, adv = pernefri_note(ktv)
        st.subheader(f"Kt/V = **{ktv}**")
        st.write(f"Status: **{status}**")
        if adv:
            st.warning("Rekomendasi:")
            for a in adv:
                st.write("- " + a)

# -------------------------------------
# KANAN — FOOD PHOTO RECOGNITION
# -------------------------------------
with col2:
    st.header("📸 Analisis Foto Makanan")

    up = st.file_uploader("Upload foto makanan", type=["jpg","jpeg","png"])

    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Foto", use_container_width=True)

        label = analyze_food(img)
        st.success(f"Terdeteksi: **{label}**")

        grams = st.number_input("Perkiraan berat makanan (gram)", value=150, step=10)

        if st.button("Tambah ke log makanan"):
            add_food(label, grams)
            st.success("Ditambahkan ke log!")

# -------------------------------------
# LOG MAKANAN
# -------------------------------------
st.markdown("---")
st.header("📒 Log Asupan Harian")

if st.session_state.food_log:
    df = pd.DataFrame(st.session_state.food_log)
    st.dataframe(df)

    totals = df[["kcal","protein","fat","carb","k","p","ca"]].sum()
    st.subheader("Total Nutrisi Hari Ini")
    st.json({
        "Energi (kcal)": float(totals["kcal"]),
        "Protein (g)": float(totals["protein"]),
        "Lemak (g)": float(totals["fat"]),
        "Karbohidrat (g)": float(totals["carb"]),
        "Kalium (mg)": float(totals["k"]),
        "Fosfat (mg)": float(totals["p"]),
        "Kalsium (mg)": float(totals["ca"]),
    })
else:
    st.info("Belum ada makanan yang dicatat hari ini.")

