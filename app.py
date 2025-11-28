# app.py — single-file app: Kt/V + heuristic food-photo + FAO sync
import streamlit as st
from PIL import Image
import io
import math
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import tempfile
import os
import difflib

# ---------------------------
# Config / small nutrition DB (default local)
# ---------------------------
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

# ---------------------------
# Simple heuristics for color/texture prototypes
# ---------------------------
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

# ---------------------------
# Utilities (image quantization / texture)
# ---------------------------
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
    N = len(pixels)
    centers = []
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

# ---------------------------
# Analyzer (heuristic)
# ---------------------------
def analyze_food_image(image: Image.Image, n_clusters=4):
    img_np = _image_to_np(image, max_side=500)
    H,W,_ = img_np.shape
    gray = np.array(image.convert("L").resize((W,H)))

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
    est_total_grams = 450.0
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
# Remote nutrition DB sync (FAO / INFOODS) + fuzzy matching
# ---------------------------
FAO_INFOODS_URL = "https://openknowledge.fao.org/bitstreams/87db546a-3e77-4f95-81a5-ecf3ab6232c7/download"

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_and_parse_fao_infoods(url=FAO_INFOODS_URL):
    """Download FAO INFOODS Excel and parse into a mapping."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    # save to temp file
    suffix = ".xlsx" if url.lower().endswith("xlsx") else ".xls"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    try:
        xls = pd.ExcelFile(tmp_path)
        dfs = []
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl' if tmp_path.endswith('.xlsx') else None)
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            raise RuntimeError("Tidak ada sheet terbaca dari file FAO.")

        df = max(dfs, key=lambda d: d.shape[0]*d.shape[1])
        cols = [c.lower() for c in df.columns]

        def find_col(cols_list, candidates):
            for cand in candidates:
                for c in cols_list:
                    if cand in c:
                        return c
            return None

        name_col = find_col(cols, ["food", "item", "description", "name"])
        if name_col is None:
            name_col = cols[0]

        energy_col = find_col(cols, ["energy", "kcal", "calori"])
        protein_col = find_col(cols, ["protein"])
        fat_col = find_col(cols, ["fat", "lipid"])
        carb_col = find_col(cols, ["carbohydrate", "carb", "carbohydrates"])
        potassium_col = find_col(cols, ["potassium", "k+", "kalium"])
        phosphorus_col = find_col(cols, ["phosphor", "phosphorus", "phosphate"])
        calcium_col = find_col(cols, ["calcium"])

        mapping = {}
        for _, row in df.iterrows():
            try:
                name = str(row[name_col]).strip()
                if not name or name.lower().startswith("nan"):
                    continue
                def safe_val(col):
                    try:
                        return float(row[col])
                    except Exception:
                        return None
                kcal = safe_val(energy_col) or 0.0
                protein = safe_val(protein_col) or 0.0
                fat = safe_val(fat_col) or 0.0
                carb = safe_val(carb_col) or 0.0
                potassium = safe_val(potassium_col) or 0.0
                phosphorus = safe_val(phosphorus_col) or 0.0
                calcium = safe_val(calcium_col) or 0.0

                key = name.lower()
                mapping[key] = {
                    "kcal": round(kcal, 2),
                    "protein": round(protein, 2),
                    "fat": round(fat, 2),
                    "carb": round(carb, 2),
                    "potassium": round(potassium, 2),
                    "phosphate": round(phosphorus, 2),
                    "calcium": round(calcium, 2),
                }
            except Exception:
                continue
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not mapping:
        raise RuntimeError("Parsing FAO file gagal — tidak ada mapping nutrisi dibuat.")
    return mapping

def fuzzy_match_food(name, remote_keys, cutoff=0.6):
    if not name:
        return None, 0.0
    name = name.lower()
    if name in remote_keys:
        return name, 1.0
    matches = difflib.get_close_matches(name, remote_keys, n=3, cutoff=cutoff)
    if matches:
        best = matches[0]
        score = difflib.SequenceMatcher(a=name, b=best).ratio()
        return best, score
    tokens = set(name.split())
    best = None; best_score = 0.0
    for k in remote_keys:
        overlap = len(tokens.intersection(set(k.split())))
        score = overlap / max(1, len(tokens))
        if score > best_score:
            best_score = score; best = k
    if best_score > 0:
        return best, best_score
    return None, 0.0

@st.cache_data(show_spinner=False)
def build_nutrition_lookup(remote_mapping=None, local_db=None):
    result = {}
    remote_keys = set(remote_mapping.keys()) if remote_mapping else set()
    local_keys = set(local_db.keys()) if local_db else set()

    for lk in local_keys:
        normalized = lk.lower()
        if remote_mapping:
            match, score = fuzzy_match_food(normalized, remote_keys, cutoff=0.6)
            if match:
                rem = remote_mapping[match]
                result[lk] = {
                    "kcal": rem.get("kcal", 0.0),
                    "protein": rem.get("protein", 0.0),
                    "fat": rem.get("fat", 0.0),
                    "carb": rem.get("carb", 0.0),
                    "potassium": rem.get("potassium", 0.0),
                    "phosphate": rem.get("phosphate", 0.0),
                    "calcium": rem.get("calcium", 0.0),
                    "_source": f"FAO:match({match},score={score:.2f})"
                }
                continue
        if local_db and lk in local_db:
            v = local_db[lk]
            result[lk] = {
                "kcal": v.get("kcal", 0.0),
                "protein": v.get("protein", 0.0),
                "fat": v.get("fat", 0.0),
                "carb": v.get("carb", 0.0),
                "potassium": v.get("potassium", 0.0),
                "phosphate": v.get("phosphate", 0.0),
                "calcium": v.get("calcium", 0.0),
                "_source": "local"
            }
    for rk in remote_keys:
        if rk not in [k.lower() for k in local_keys]:
            rem = remote_mapping[rk]
            result[rk] = {
                "kcal": rem.get("kcal", 0.0),
                "protein": rem.get("protein", 0.0),
                "fat": rem.get("fat", 0.0),
                "carb": rem.get("carb", 0.0),
                "potassium": rem.get("potassium", 0.0),
                "phosphate": rem.get("phosphate", 0.0),
                "calcium": rem.get("calcium", 0.0),
                "_source": "FAO_remote"
            }
    return result

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Kt/V + Food Photo (heuristic)", layout="wide")
st.title("Kt/V + Food Photo — Heuristic (no API)")
# minimal dependency notice
st.sidebar.header("Panduan singkat")
st.sidebar.write("""
- Analisis foto: heuristic segmen warna + tekstur. Koreksi manual disarankan.
- Sinkronisasi nutrisi menggunakan FAO/INFOODS (opsional).
- Untuk akurasi klinis, gunakan model terlatih atau Vision API.
""")

# ---------------------------
# Sidebar: Sync nutrition DB
# ---------------------------
with st.sidebar.expander("🔄 Sync database nutrisi (FAO INFOODS)"):
    st.write("Sinkronisasi database nutrisi dari FAO/INFOODS (public).")
    if st.button("Sync nutrisi dari FAO (INFOODS)"):
        try:
            remote = fetch_and_parse_fao_infoods()
            merged = build_nutrition_lookup(remote_mapping=remote, local_db=FOOD_DB_PER_100G)
            st.success(f"Sinkronisasi berhasil. Remote: {len(remote)} item. Merged: {len(merged)} item.")
            sample_keys = list(merged.keys())[:15]
            st.dataframe(pd.DataFrame({k: merged[k] for k in sample_keys}).T)
            if st.checkbox("Overwrite local FOOD_DB_PER_100G (destructive)"):
                for k,v in merged.items():
                    FOOD_DB_PER_100G[k] = {
                        "kcal": float(v.get("kcal",0.0)),
                        "protein": float(v.get("protein",0.0)),
                        "fat": float(v.get("fat",0.0)),
                        "carb": float(v.get("carb",0.0)),
                        "potassium": float(v.get("potassium",0.0)),
                        "phosphate": float(v.get("phosphate",0.0)),
                        "calcium": float(v.get("calcium",0.0))
                    }
                st.success("Local FOOD_DB_PER_100G diperbarui!")
        except Exception as e:
            st.error(f"Gagal sync: {e}. Jika hosting memblokir unduh, unggah file Excel/CSV memakai upload di bawah.")
    st.markdown("---")
    uploaded_db = st.file_uploader("Unggah file Excel/CSV komposisi pangan (opsional)", type=["csv","xlsx"])
    if uploaded_db is not None:
        try:
            if uploaded_db.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_db)
            else:
                df_upload = pd.read_excel(uploaded_db)
            st.success(f"File terbaca: {df_upload.shape[0]} baris")
            st.dataframe(df_upload.head(10))
        except Exception as e:
            st.error(f"Gagal membaca file uploaded: {e}")

# ---------------------------
# Main layout
# ---------------------------
left, right = st.columns([1,1])

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

with right:
    st.header("📸 Analisis Foto Makanan — Food Diary")
    st.markdown("Unggah foto makanan (jpg/png) — sistem akan menebak komponen, koreksi label/porsi, dan simpan ke diary harian.")
    uploaded = st.file_uploader("Unggah foto makanan", type=["jpg","jpeg","png"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Preview", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
            image = None

        if image is not None:
            st.info("Menganalisis (heuristic)...")
            analysis = analyze_food_image(image, n_clusters=4)
            detected = analysis.get("detected", [])
            if not detected:
                st.warning("Tidak terdeteksi item — coba foto lain (lebih terang/kontras).")
            else:
                st.markdown("### Deteksi (koreksi bila perlu)")
                entries = []
                for i,d in enumerate(detected):
                    st.write(f"**Item #{i+1}** — tebakan: **{d['label']}** (area {d['area_frac']:.2%})")
                    cols = st.columns([2,1,1,1,1])
                    cols[0].markdown(f"<div style='width:36px;height:18px;background:rgb{tuple(d['color'])};border:1px solid #444'></div>", unsafe_allow_html=True)
                    sel_label = cols[0].selectbox(f"Label #{i+1}", options=FOOD_KEYS, index=FOOD_KEYS.index(d['label']) if d['label'] in FOOD_KEYS else 0, key=f"label_{i}")
                    portion = cols[1].number_input(f"Porsi (g) #{i+1}", min_value=10, max_value=2000, value=int(d["portion_g"]), step=10, key=f"portion_{i}")
                    per100 = FOOD_DB_PER_100G.get(sel_label, FOOD_DB_PER_100G["unknown"])
                    factor = portion / 100.0
                    entry = {
                        "datetime": datetime.utcnow().isoformat(),
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
                    entries.append(entry)

                if st.button("Simpan ke diary"):
                    # load existing diary from session_state
                    diary = st.session_state.get("food_diary", [])
                    diary.extend(entries)
                    st.session_state["food_diary"] = diary
                    st.success(f"{len(entries)} item disimpan ke diary (session).")

    st.markdown("### Food diary hari ini")
    diary = st.session_state.get("food_diary", [])
    if diary:
        df_diary = pd.DataFrame(diary)
        # aggregate daily totals by date (UTC date)
        df_diary['date'] = pd.to_datetime(df_diary['datetime']).dt.date
        today = datetime.utcnow().date()
        today_df = df_diary[df_diary['date'] == today]
        st.write(f"Entri hari ini: {len(today_df)}")
        if not today_df.empty:
            agg = {
                "Energi (kcal)": today_df["kcal"].sum(),
                "Protein (g)": today_df["protein_g"].sum(),
                "Lemak (g)": today_df["fat_g"].sum(),
                "Karbohidrat (g)": today_df["carb_g"].sum(),
                "Kalium (mg)": today_df["potassium_mg"].sum(),
                "Fosfat (mg)": today_df["phosphate_mg"].sum(),
                "Kalsium (mg)": today_df["calcium_mg"].sum(),
            }
            st.table(pd.DataFrame.from_dict(agg, orient='index', columns=['Total']))
        st.dataframe(df_diary.sort_values('datetime', ascending=False).reset_index(drop=True))

        csv = df_diary.to_csv(index=False).encode('utf-8')
        st.download_button("Download diary (CSV)", data=csv, file_name="food_diary.csv", mime="text/csv")
    else:
        st.info("Diary kosong — simpan foto makanan untuk mulai.")

st.markdown("---")
st.caption("⚠ Prototype: heuristic. Gunakan koreksi manual setelah scan. Untuk keputusan klinis, berkonsultasilah dengan tim medis.")
