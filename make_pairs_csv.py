# make_pairs_csv_beginner.py
# ------------------------------------------------------------
# PURPOSE
#   Create a CSV that pairs every IMAGE slice with its MASK slice.
#   It assumes your structure looks like:
#     ROOT/<case>-t1c/vol/images/*.png
#     ROOT/<case>-t1n/vol/images/*.png
#     ROOT/<case>-t2f/vol/images/*.png
#     ROOT/<case>-t2w/vol/images/*.png
#     ROOT/<case>-seg/vol/images/*.png
#   and filenames like:
#     <case>-t1c_vol_axial_077.png  <->  <case>-seg_vol_axial_077.png
#
# OUTPUT
#   pairs.csv with columns:
#     case_id, modality, slice_idx, image_path, mask_path, mask_exists
#
# HOW TO RUN
#   python make_pairs_csv_beginner.py
# ------------------------------------------------------------

import os, re, csv

# ====== EDIT THIS ======
ROOT_DIR = "/Users/arunyahooda/Desktop/BT2O23/Training_png"
OUT_CSV  = "/Users/arunyahooda/Desktop/BT2O23/train_pairs"
# =======================

# Your modality tokens as they appear in folder/file names
MOD_TOKENS = ["-t1c", "-t1n", "-t2f", "-t2w"]  # adjust if you have others

# Regex to grab the 3-digit slice index at the end: ..._077.png
SLICE_RE = re.compile(r"_(\d{3})\.png$", re.IGNORECASE)

def is_png(p): return p.lower().endswith(".png")

def walk_image_dirs(root):
    """
    Yield (dirpath, files) for image folders that look like:
      .../<case>-<mod>/vol/images
    and are NOT the '-seg' mask folder.
    """
    for cur, _, files in os.walk(root):
        # we only want folders named 'images'
        if os.path.basename(cur).lower() != "images":
            continue
        # skip mask folders
        if "-seg" in cur.lower():
            continue
        # must contain pngs
        pngs = [f for f in files if is_png(f)]
        if not pngs:
            continue
        yield cur, sorted(pngs)

def parse_case_and_mod(dirpath):
    """
    From: ROOT/<case>-<mod>/vol/images
    Return: (case_id, modality_token)
    """
    # parent: .../<case>-<mod>/vol/images  -> get <case>-<mod>
    parent = os.path.dirname(os.path.dirname(dirpath))
    case_mod = os.path.basename(parent)   # e.g., BraTS-GLI-00000-000-t1c
    # find which mod token is present
    mod = None
    for tok in MOD_TOKENS:
        if case_mod.lower().endswith(tok):
            mod = tok.lstrip("-")
            case_id = case_mod[: -len(tok)]
            break
    if mod is None:
        # fallback: treat anything not ending with -seg as 'vol'
        mod = "vol"
        case_id = case_mod
    return case_id, mod

def build_mask_path_for_image(image_dir, case_id, slice_name):
    """
    Given an image folder path and the case/modality info, derive the
    corresponding mask filename & path. We replace the modality part
    with '-seg' and keep the rest (including 'vol_axial_077.png').
    """
    # image_dir: ROOT/<case>-t1c/vol/images
    # mask_dir:  ROOT/<case>-seg/vol/images
    case_mod_dir = os.path.dirname(os.path.dirname(image_dir))  # ROOT/<case>-t1c
    root = os.path.dirname(case_mod_dir)                        # ROOT
    mask_dir = os.path.join(root, f"{case_id}-seg", "vol", "images")

    # slice_name example: BraTS-GLI-00000-000-t1c_vol_axial_077.png
    # mask file should be:  BraTS-GLI-00000-000-seg_vol_axial_077.png
    mask_name = slice_name.replace(f"{case_id}-t1c", f"{case_id}-seg") \
                          .replace(f"{case_id}-t1n", f"{case_id}-seg") \
                          .replace(f"{case_id}-t2f", f"{case_id}-seg") \
                          .replace(f"{case_id}-t2w", f"{case_id}-seg")

    return os.path.join(mask_dir, mask_name)

def main():
    rows = []
    total = 0
    missing_masks = 0

    for img_dir, files in walk_image_dirs(ROOT_DIR):
        case_id, modality = parse_case_and_mod(img_dir)

        for fname in files:
            total += 1
            # pull 3-digit index at end (077)
            m = SLICE_RE.search(fname)
            slice_idx = m.group(1) if m else "NA"

            img_path = os.path.join(img_dir, fname)
            mask_path = build_mask_path_for_image(img_dir, case_id, fname)
            mask_exists = os.path.exists(mask_path)

            if not mask_exists:
                missing_masks += 1

            rows.append([case_id, modality, slice_idx, img_path, mask_path, int(mask_exists)])

    # write CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "modality", "slice_idx", "image_path", "mask_path", "mask_exists"])
        w.writerows(rows)

    print(f"Total image slices: {total}")
    print(f"Pairs written to:   {OUT_CSV}")
    print(f"Missing masks:      {missing_masks}")

if __name__ == "__main__":
    main()
