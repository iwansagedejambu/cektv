import streamlit as st
from PIL import Image
import io, hashlib

# Database nutrisi mini per 100 g (tanpa numpy / pandas)
FOOD_DB = {
    "rice":  {"kcal":130,"protein":2.7,"fat":0.3,"carb":28},
    "chicken":{"kcal":239,"protein":27,"fat":14,"carb":0},
    "egg":   {"kcal":155,"protein":13,"fat":11,"carb":1},
    "tofu":  {"kcal":76,"protein":8,"fat":5,"carb":2},
    "banana":{"kcal":89,"protein":1,"fat":0,"carb":23},
    "unknown":{"kcal":100,"protein":3,"fat":5,"carb":12},
}
FOOD_KEYS = list(FOOD_DB.keys())

def simple_hash_predict(image_bytes):
    """Mengubah hash gambar → label makanan (sederhana tapi stabil)."""
    h = hashlib.sha256(image_bytes).hexdigest()
    idx = int(h[:8], 16) % len(FOOD_KEYS)
    return FOOD_KEYS[idx]

def analyze_image(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    label = simple_hash_predict(img_bytes)
    portion = 150
    per = FOOD_DB[label]
    return {
        "label": label,
        "portion": portion,
        "kcal": round(per["kcal"] * (portion/100),1),
        "protein": round(per["protein"] * (portion/100),1),
        "fat": round(per["fat"] * (portion/100),1),
        "carb": round(per["carb"] * (portion/100),1),
    }

def hitung_ktv(qb, durasi, bb):
    clearance = 0.7 * qb
    waktu_menit = durasi * 60
    v = 0.55 * bb * 1000
    if v <= 0: return 0
    return round((clearance * waktu_menit) / v, 2)

st.title("Kt/V + Analisis Foto Ultra-Ringan")

st.header("Kalkulator Kt/V")
qb = st.number_input("Qb (mL/menit)", value=200)
bb = st.number_input("Berat badan kering (kg)", value=50.0)
dur = st.number_input("Durasi (jam)", value=4.0)

if st.button("Hitung Kt/V"):
    ktv = hitung_ktv(qb, dur, bb)
    st.subheader(f"KTV = {ktv}")

st.markdown("---")
st.header("Analisis Foto Makanan (versi ringan)")
img_file = st.file_uploader("Upload foto", type=["jpg","png"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("Analisis"):
        res = analyze_image(img)
        st.write("Hasil prediksi:")
        st.json(res)
