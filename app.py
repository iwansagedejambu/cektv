import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime

# =====================================
# CONFIG HALAMAN
# =====================================
st.set_page_config(page_title="Kt/V & Food Tracker (USDA)", layout="wide")

# =====================================
# 1) FUNGSI KALKULATOR Kt/V
# =====================================

def hitung_ktv(qb, durasi_jam, bb_kering):
    """
    qb          : aliran darah (mL/menit)
    durasi_jam  : lama HD (jam)
    bb_kering   : berat badan kering (kg)
    """
    clearance = 0.7 * qb        # asumsi clearance
    waktu_menit = durasi_jam * 60
    v = 0.55 * bb_kering * 1000 # volume distribusi urea (mL)
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
# 2) LOAD DATABASE USDA (3 FILE UTAMA)
#    food.csv, nutrient.csv, food_nutrient.csv
# =====================================

@st.cache_data
def load_usda_simple():
    """
    Menggabungkan 3 file USDA:
    - data/food.csv           -> fdc_id, description
    - data/nutrient.csv       -> id, name, unit_name
    - data/food_nutrient.csv  -> fdc_id, nutrient_id, amount

    Output: dataframe sederhana per 100 g dengan kolom:
    [fdc_id, description, kcal, protein, fat, carb, k, p, ca]
    """
    try:
        food = pd.read_csv("data/food.csv", usecols=["fdc_id", "description"])
        nutrient = pd.read_csv("data/nutrient.csv", usecols=["id", "name", "unit_name"])
        food_nutrient = pd.read_csv(
            "data/food_nutrient.csv",
            usecols=["fdc_id", "nutrient_id", "amount"]
        )
    except Exception as e:
        st.error(f"Gagal membaca file USDA. Pastikan file ada di folder 'data/'.\nDetail: {e}")
        return pd.DataFrame()

    # Nutrisinya mana saja yang mau dipakai?
    target_names = {
        "Energy": "kcal",
        "Protein": "protein",
        "Total lipid (fat)": "fat",
        "Carbohydrate, by difference": "carb",
        "Potassium, K": "k",
        "Phosphorus, P": "p",
        "Calcium, Ca": "ca",
    }

    nutr_filt = nutrient[nutrient["name"].isin(target_names.keys())]

    fn_filt = food_nutrient[
        food_nutrient["nutrient_id"].isin(nutr_filt["id"])
    ]

    merged = fn_filt.merge(
        nutr_filt,
        left_on="nutrient_id",
        right_on="id",
        how="left"
    )

    merged = merged.merge(
        food,
        on="fdc_id",
        how="left"
    )

    pivot = merged.pivot_table(
        index=["fdc_id", "description"],
        columns="name",
        values="amount",
        aggfunc="mean"
    )

    pivot = pivot.rename(columns=target_names).reset_index()

    return pivot

USDA_SIMPLE = load_usda_simple()

# =====================================
# 3) FUNGSI NUTRISI DARI USDA
# =====================================

def get_nutrition_from_usda_by_desc(desc_keyword: str, grams: float):
    """
    Mencari makanan di USDA_SIMPLE yang 'description'-nya
    mengandung desc_keyword (case-insensitive), lalu
    mengembalikan nutrisi untuk sejumlah 'grams'.
    """
    if USDA_SIMPLE.empty:
        return None

    df = USDA_SIMPLE[
        USDA_SIMPLE["description"].str.contains(desc_keyword, case=False, na=False)
    ]

    if df.empty:
        return None

    row = df.iloc[0]  # ambil yang pertama dulu

    factor = grams / 100.0

    def safe_get(col):
        return round(float(row.get(col, 0)) * factor, 1)

    return {
        "description": row["description"],
        "kcal": safe_get("kcal"),
        "protein": safe_get("protein"),
        "fat": safe_get("fat"),
        "carb": safe_get("carb"),
        "k": safe_get("k"),
        "p": safe_get("p"),
        "ca": safe_get("ca"),
    }

# =====================================
# 4) KLASSIFIKASI FOTO MAKANAN (SANGAT SEDERHANA)
# =====================================

# prototipe warna rata-rata
PROTOTYPES = {
    "rice": (245, 245, 240),
    "chicken": (220, 180, 150),
    "egg": (250, 230, 140),
    "tofu": (245, 245, 235),
    "banana": (240, 210, 60),
    "salmon": (230, 120, 120),
    "mixed_salad": (80, 150, 80),
}

def rgb_dist(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5

def analyze_food(image: Image.Image):
    """
    Analisis warna rata-rata gambar untuk menebak jenis makanan.
    Ini hanya dummy sederhana, nanti bisa diganti model ML beneran.
    """
    img = image.resize((200, 200))
    arr = np.array(img)
    pixels = arr.reshape(-1, 3)
    mean_color = pixels.mean(axis=0)

    best = None
    best_score = 999999
    for name, proto in PROTOTYPES.items():
        d = rgb_dist(mean_color, proto)
        if d < best_score:
            best_score = d
            best = name
    return best or "unknown"

# mapping label → kata kunci description USDA
FOOD_LABEL_TO_USDA_DESC = {
    "rice": "rice, white, cooked",
    "chicken": "chicken, breast, cooked",
    "egg": "egg, whole, boiled",
    "tofu": "tofu, firm",
    "banana": "banana, raw",
    "salmon": "salmon, atlantic",
    "mixed_salad": "salad",
    "unknown": "",  # nanti fallback default
}

# nilai default kalau USDA tidak ketemu
DEFAULT_NUT = {
    "description": "Unknown food",
    "kcal": 100,
    "protein": 3,
    "fat": 5,
    "carb": 12,
    "k": 50,
    "p": 40,
    "ca": 20,
}

# =====================================
# 5) FOOD LOG DI SESSION
# =====================================

if "food_log" not in st.session_state:
    st.session_state.food_log = []

def add_food(label, grams):
    """
    Menambahkan makanan ke log dengan menggunakan data USDA.
    """
    # cari keyword description dari label
    keyword = FOOD_LABEL_TO_USDA_DESC.get(label, label)

    if keyword:
        nut = get_nutrition_from_usda_by_desc(keyword, grams)
    else:
        nut = None

    if nut is None:
        st.warning("Data USDA untuk makanan ini tidak ditemukan, pakai nilai default.")
        nut = DEFAULT_NUT.copy()
        nut["description"] = label

    entry = {
        "time": datetime.now().strftime("%H:%M"),
        "food": nut["description"],
        "label": label,
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
# 6) UI — DUA KOLOM: Kt/V & FOTO MAKANAN
# =====================================

col1, col2 = st.columns([1, 1])

# ---- KIRI: KALKULATOR Kt/V ----
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

# ---- KANAN: FOTO MAKANAN ----
with col2:
    st.header("📸 Analisis Foto Makanan (USDA)")

    up = st.file_uploader("Upload foto makanan", type=["jpg", "jpeg", "png"])

    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Foto yang diupload")  # paling simpel

        label = analyze_food(img)
        st.success(f"Perkiraan jenis makanan: **{label}**")

        grams = st.number_input(
            "Perkiraan berat makanan (gram)", value=150, step=10, key="grams_photo"
        )

        if st.button("Tambah ke log makanan dari foto"):
            add_food(label, grams)
            st.success("Makanan ditambahkan ke log!")

# =====================================
# 7) OPSIONAL: PILIH LANGSUNG DARI USDA
# =====================================

st.markdown("---")
st.subheader("🔍 Tambah dari daftar USDA (tanpa foto)")

if not USDA_SIMPLE.empty:
    selected_desc = st.selectbox(
        "Cari & pilih makanan dari USDA",
        options=USDA_SIMPLE["description"].sort_values().unique()
    )
    grams_manual = st.number_input(
        "Berat (gram) untuk makanan di atas",
        value=100,
        step=10,
        key="grams_usda"
    )

    if st.button("Tambah ke log dari daftar USDA"):
        nut = get_nutrition_from_usda_by_desc(selected_desc, grams_manual)
        if nut is None:
            st.warning("Gagal membaca nutrisi dari USDA, pakai nilai default.")
            nut = DEFAULT_NUT.copy()
            nut["description"] = selected_desc

        entry = {
            "time": datetime.now().strftime("%H:%M"),
            "food": nut["description"],
            "label": "manual_usda",
            "grams": grams_manual,
            "kcal": nut["kcal"],
            "protein": nut["protein"],
            "fat": nut["fat"],
            "carb": nut["carb"],
            "k": nut["k"],
            "p": nut["p"],
            "ca": nut["ca"],
        }
        st.session_state.food_log.append(entry)
        st.success("Makanan dari USDA ditambahkan ke log!")
else:
    st.info("USDA_SIMPLE kosong. Cek lagi file CSV di folder 'data/'.")

# =====================================
# 8) TABEL LOG MAKANAN & TOTAL NUTRISI
# =====================================

st.markdown("---")
st.header("📒 Log Asupan Harian")

if st.session_state.food_log:
    df = pd.DataFrame(st.session_state.food_log)
    st.dataframe(df)

    totals = df[["kcal", "protein", "fat", "carb", "k", "p", "ca"]].sum()
    st.subheader("Total Nutrisi Hari Ini (perkiraan)")
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
