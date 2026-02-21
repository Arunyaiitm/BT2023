# make_rl_splits_t1c.py
# ------------------------------------------------------------
# PURPOSE
#   From your big train_pairs.csv, create the paper-style subset:
#     - modality = T1CE only (we assume it's named "t1c" in your CSV)
#     - choose ONE lesion-containing slice per case (mask > 0)
#     - take 30 for TRAIN and 30 for TEST
#   Save two small CSVs you can use for the RL experiments.
#
# HOW TO RUN
#   1) pip install pandas pillow numpy
#   2) python make_rl_splits_t1c.py
#
# WHAT IT WRITES
#   - rl_train_t1c_30.csv  (30 rows)
#   - rl_test_t1c_30.csv   (30 rows)
#
# COLUMNS KEPT
#   case_id, modality, slice_idx, image_path, mask_path
# ------------------------------------------------------------

import os, random
import numpy as np
import pandas as pd
from PIL import Image

# ====== EDIT THESE IF NEEDED ======
CSV_PATH = "/Users/arunyahooda/Desktop/BT2O23/train_pairs.csv"  # your big manifest
OUT_DIR  = "/Users/arunyahooda/Desktop/BT2O23"                  # where to save the 2 small CSVs
MODALITY = "t1c"       # T1 post-contrast (paper uses this)
N_TRAIN  = 30
N_TEST   = 30
RNG_SEED = 123
# ==================================

# (Optional) expected image size; if set, we enforce it
EXPECTED_SHAPE = (240, 240)   # height, width
ENFORCE_SHAPE  = True         # set False to skip shape checking

def load_gray_u8(path):
    """Open a PNG and return a grayscale uint8 numpy array."""
    with Image.open(path) as im:
        return np.array(im.convert("L"), dtype=np.uint8)

def has_lesion(mask_path):
    """True if the mask has any non-zero pixels."""
    m = load_gray_u8(mask_path)
    if ENFORCE_SHAPE and m.shape != EXPECTED_SHAPE:
        return False
    return (m > 0).any()

def pick_one_slice_with_lesion(rows):
    """
    Given a DataFrame (all slices for one case),
    return a single row where mask > 0. If none, return None.
    """
    # Shuffle rows to avoid always picking the same slice index
    rows = rows.sample(frac=1.0, random_state=RNG_SEED)
    for _, r in rows.iterrows():
        if os.path.exists(r["mask_path"]) and has_lesion(r["mask_path"]):
            return r
    return None

def main():
    # 1) Load CSV
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 2) Basic filtering: modality and masks must exist
    if "mask_exists" in df.columns:
        df = df[df["mask_exists"] == 1]
    df = df[df["modality"].str.lower() == MODALITY.lower()].copy()

    if df.empty:
        raise RuntimeError(f"No rows found for modality '{MODALITY}' in {CSV_PATH}")

    # 3) Group by case and pick ONE lesion slice per case
    selected_rows = []
    for case_id, g in df.groupby("case_id"):
        picked = pick_one_slice_with_lesion(g)
        if picked is not None:
            selected_rows.append(picked)

    if len(selected_rows) < (N_TRAIN + N_TEST):
        raise RuntimeError(
            f"Not enough cases with lesion in modality '{MODALITY}'. "
            f"Found {len(selected_rows)}, need {N_TRAIN + N_TEST}."
        )

    # 4) Shuffle cases and split into train/test
    random.Random(RNG_SEED).shuffle(selected_rows)
    train_rows = selected_rows[:N_TRAIN]
    test_rows  = selected_rows[N_TRAIN:N_TRAIN + N_TEST]

    # 5) Build tidy DataFrames with only the columns we need
    keep_cols = ["case_id", "modality", "slice_idx", "image_path", "mask_path"]
    train_df = pd.DataFrame(train_rows)[keep_cols].reset_index(drop=True)
    test_df  = pd.DataFrame(test_rows)[keep_cols].reset_index(drop=True)

    # 6) Save
    os.makedirs(OUT_DIR, exist_ok=True)
    train_csv = os.path.join(OUT_DIR, "rl_train_t1c_30.csv")
    test_csv  = os.path.join(OUT_DIR, "rl_test_t1c_30.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # 7) Print a friendly summary
    print(f"Modality:          {MODALITY}")
    print(f"Cases with lesion: {len(selected_rows)}")
    print(f"Train slices:      {len(train_df)} -> {train_csv}")
    print(f"Test slices:       {len(test_df)}  -> {test_csv}")

    # Peek a couple rows so you see the format
    print("\nSample TRAIN rows:")
    print(train_df.head(3).to_string(index=False))
    print("\nSample TEST rows:")
    print(test_df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
