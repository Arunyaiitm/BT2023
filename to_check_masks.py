# png_leaf_inspector_beginner.py
# ------------------------------------------------------------
# PURPOSE
#   Your previous script printed "Cases checked: 0" because it only looked
#   ONE level deep. This beginner-friendly script scans *recursively* and
#   inspects every folder that actually contains PNG files (at any depth).
#
# WHAT IT DOES
#   - Walks ROOT_DIR recursively
#   - For each folder that directly contains PNGs, it:
#       * picks the middle PNG by name,
#       * converts it to grayscale,
#       * counts unique pixel values,
#       * classifies the folder as LIKELY_IMAGE or LIKELY_MASK
#         (mask folders often have few unique values like 0 and 255).
#   - Prints a summary and a small preview table.
#
# EDIT THESE:
#   1) ROOT_DIR: the parent directory you want to scan (e.g., Training_png)
#   2) UNIQUE_THRESHOLD: <= this many unique values => treat as mask
#   3) SHOW_FIRST: how many rows to show from the report
#
# RUN:
#   python png_leaf_inspector_beginner.py
# ------------------------------------------------------------

import os
from typing import List, Tuple
import numpy as np
from PIL import Image

# ====== EDIT THESE ======
ROOT_DIR = "/Users/arunyahooda/Desktop/BT2O23/Training_png"
UNIQUE_THRESHOLD = 10    # <= 10 unique grayscale values -> LIKELY_MASK
SHOW_FIRST = 12          # show this many result rows
# ========================


def folders_with_pngs(root: str) -> List[str]:
    """
    Walk recursively and collect every folder that directly contains
    at least one .png file. (We don't care about folder names here.)
    """
    result = []
    for cur_dir, _, files in os.walk(root):
        pngs = [f for f in files if f.lower().endswith(".png")]
        if pngs:
            result.append(cur_dir)
    return sorted(result)


def pick_middle_png(dir_path: str) -> str:
    """
    Return the full path to the 'middle' PNG in this folder
    (based on sorted filename order). Assumes the folder has PNGs.
    """
    files = sorted(f for f in os.listdir(dir_path) if f.lower().endswith(".png"))
    mid = len(files) // 2
    return os.path.join(dir_path, files[mid])


def classify_by_pixels(png_path: str, unique_threshold: int) -> Tuple[str, Tuple[int, int], int, List[int]]:
    """
    Open one PNG, convert to grayscale, count unique values,
    and classify as LIKELY_MASK or LIKELY_IMAGE.
    Returns: (kind, shape, unique_count, first_values_list)
    """
    with Image.open(png_path) as im:
        gray = im.convert("L")        # grayscale
        arr = np.array(gray)          # HxW uint8

    uniques = np.unique(arr)
    unique_count = int(len(uniques))
    # Small number of unique values => likely a label mask (e.g., {0, 255})
    kind = "LIKELY_MASK" if unique_count <= unique_threshold else "LIKELY_IMAGE"

    # Show up to first 10 unique values for debugging/inspection
    first_values = uniques[:10].tolist()

    return kind, arr.shape, unique_count, first_values


def main():
    # 1) Find all leaf folders that contain PNGs
    leaf_dirs = folders_with_pngs(ROOT_DIR)

    if not leaf_dirs:
        print("No PNG-containing folders found. Check ROOT_DIR or run your converter first.")
        return

    report = []
    total_pngs_seen = 0

    # 2) Inspect one representative PNG per folder
    for d in leaf_dirs:
        try:
            sample_png = pick_middle_png(d)
        except Exception:
            # If something odd happens (empty folder race condition), skip it
            continue

        # Optional hint from folder name
        name_hint = "MASK_DIR" if "mask" in d.lower() else ("IMAGE_DIR" if "image" in d.lower() else "UNKNOWN_DIR")

        # Pixel-based classification
        kind, shape, uniq_count, first_vals = classify_by_pixels(sample_png, UNIQUE_THRESHOLD)

        # Count how many PNGs are in this folder (for your info)
        num_pngs = sum(1 for f in os.listdir(d) if f.lower().endswith(".png"))
        total_pngs_seen += num_pngs

        report.append({
            "dir": d,
            "name_hint": name_hint,          # based on folder name only
            "kind": kind,                    # based on pixel uniqueness
            "sample_png": os.path.basename(sample_png),
            "shape": shape,                  # (H, W)
            "unique_count": uniq_count,      # how many unique pixel values
            "first_values": first_vals,      # first 10 unique values
            "num_pngs": num_pngs,            # how many PNGs in this folder
        })

    # 3) Summary
    mask_like = sum(1 for r in report if r["kind"] == "LIKELY_MASK")
    image_like = sum(1 for r in report if r["kind"] == "LIKELY_IMAGE")

    print(f"Folders scanned (with PNGs): {len(report)}")
    print(f"Total PNG files seen:       {total_pngs_seen}")
    print(f"Classified as LIKELY_MASK:  {mask_like}")
    print(f"Classified as LIKELY_IMAGE: {image_like}\n")

    # 4) Preview rows
    print("Sample rows:")
    for r in report[:SHOW_FIRST]:
        print(
            f"- dir='{r['dir']}'\n"
            f"    name_hint={r['name_hint']}  kind={r['kind']}  pngs={r['num_pngs']}\n"
            f"    sample='{r['sample_png']}'  shape={r['shape']}  "
            f"unique_count={r['unique_count']}  first_values={r['first_values']}"
        )

    # TIP: If your PNGs are nested like ROOT/<case>/<mod>/images/*.png,
    # this script will still find 'images' folders automatically.
    # If you only want to inspect image folders (not mask folders),
    # you can filter `leaf_dirs` for paths containing "/images/" before the loop.


if __name__ == "__main__":
    main()
