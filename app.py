# ---------------------------
# Remote nutrition DB sync (FAO / INFOODS) + fuzzy matching
# ---------------------------
import requests
import tempfile
import zipfile
import os
import difflib

# FAO INFOODS sample download (AnFooD / Analytical database)
# Source (FAO Open Knowledge): has Excel bitstream; fallback to manual upload if blocked.
FAO_INFOODS_URL = "https://openknowledge.fao.org/bitstreams/87db546a-3e77-4f95-81a5-ecf3ab6232c7/download"  # (FAO AnFooD XLSX). :contentReference[oaicite:3]{index=3}

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_and_parse_fao_infoods(url=FAO_INFOODS_URL):
    """
    Download FAO INFOODS Excel and parse into a mapping:
      name_normalized -> {kcal, protein, fat, carb, potassium, phosphate (or phosphorus), calcium}
    Returns: dict
    """
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xls") as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    # Try read all sheets and find plausible nutrient columns
    try:
        xls = pd.ExcelFile(tmp_path)
        dfs = []
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl' if tmp_path.endswith('.xlsx') else None)
                dfs.append(df)
            except Exception:
                continue
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # merge candidate dfs (pick the widest one)
    if not dfs:
        raise RuntimeError("Tidak ada sheet terbaca dari file FAO.")

    # choose the largest dataframe (heuristic)
    df = max(dfs, key=lambda d: d.shape[0]*d.shape[1])

    # normalize column names to lowercase stripped
    colmap = {c: c.strip().lower() for c in df.columns}

    # helper: find a column by candidate keywords
    def find_col(df_cols, candidates):
        for cand in candidates:
            for c in df_cols:
                if cand in c:
                    return c
        return None

    cols = [c.lower() for c in df.columns]

    # candidates for food name column
    name_col = find_col(cols, ["food", "item", "description", "name"])
    if name_col is None:
        # fallback to first column
        name_col = cols[0]

    # candidates for nutrients
    energy_col = find_col(cols, ["energy", "kcal", "calori"])
    protein_col = find_col(cols, ["protein"])
    fat_col = find_col(cols, ["fat", "lipid"])
    carb_col = find_col(cols, ["carbohydrate", "carb", "carbohydrates"])
    potassium_col = find_col(cols, ["potassium", "k+", "kalium"])
    phosphorus_col = find_col(cols, ["phosphor", "phosphorus", "phosphate"])
    calcium_col = find_col(cols, ["calcium"])

    # build mapping: normalize food name -> nutrients (per 100g)
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

    if not mapping:
        raise RuntimeError("Parsing FAO file gagal — tidak ada mapping nutrisi dibuat.")

    return mapping

# fuzzy match helper: match a label to remote mapping keys
def fuzzy_match_food(name, remote_keys, cutoff=0.6):
    if not name:
        return None, 0.0
    name = name.lower()
    # exact
    if name in remote_keys:
        return name, 1.0
    matches = difflib.get_close_matches(name, remote_keys, n=3, cutoff=cutoff)
    if matches:
        # return best match and score (approx)
        best = matches[0]
        # approximate score by ratio
        score = difflib.SequenceMatcher(a=name, b=best).ratio()
        return best, score
    # fallback: token overlap heuristic
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

# function to get nutrients for a label using remote DB with fallback to local DB
@st.cache_data(show_spinner=False)
def build_nutrition_lookup(remote_mapping=None, local_db=None):
    """
    remote_mapping: dict from fetch_and_parse_fao_infoods()
    local_db: existing FOOD_DB_PER_100G (keys)
    returns: merged lookup mapping (normalized keys)
    """
    result = {}
    remote_keys = set(remote_mapping.keys()) if remote_mapping else set()
    local_keys = set(local_db.keys()) if local_db else set()

    # prefer local_db as canonical names, but try to augment from remote_mapping
    for lk in local_keys:
        normalized = lk.lower()
        # try fuzzy match to remote
        if remote_mapping:
            match, score = fuzzy_match_food(normalized, remote_keys, cutoff=0.6)
            if match:
                rem = remote_mapping[match]
                # convert remote nutrient names to same schema per 100g
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
        # fallback to existing local db values if present
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
    # Also add remote-only items (optional)
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
