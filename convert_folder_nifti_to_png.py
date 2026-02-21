# PURPOSE
#   Convert ALL BraTS-style NIfTI volumes (.nii / .nii.gz) found inside
#   your Training and Validation folders into 2D PNG slices.
#   If a matching segmentation mask "*_seg.nii(.gz)" exists next to a
#   volume (in the same case folder), matching PNG mask slices are saved.
#
# WHAT YOU EDIT
#   1) TRAIN_ROOT, VAL_ROOT  -> where your .nii/.nii.gz live
#   2) OUT_TRAIN, OUT_VAL    -> where PNGs will be saved
#   3) OPTIONS section        -> axis, resize, only tumor slices, etc.
#
# HOW TO RUN
#   python convert_brats_to_png_beginner.py
#
# OUTPUT FOLDERS (example)
#   OUT_TRAIN/
#     <case_id>/
#       <modality>/            # e.g., t1ce / flair / t2 / t1 / vol
#         images/
#            <case>_<mod>_axial_000.png
#            ...
#         masks/               # only if *_seg.nii* exists and matches shape
#            <case>_<mod>_axial_000.png
#
# NOTES
#   - We reorient volumes to RAS (common orientation) for consistent slices.
#   - We normalize each slice to 0..255 using robust contrast (1..99%).
#   - We SKIP *_seg.nii* as image inputs (those are labels), but we will
#     load them as masks for overlay/PNG export if present.
#   - If you only want slices that actually contain tumor pixels, set
#     ONLY_MASK_SLICES = True (requires masks to be present).
# ------------------------------------------------------------

import os
import re
import glob
import numpy as np
import nibabel as nib
import cv2

# ============ 1) FOLDERS TO EDIT ============

# Your BraTS Training and Validation roots (contain many case subfolders)
TRAIN_ROOT = "/Users/arunyahooda/Desktop/BT2O23/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
VAL_ROOT   = "/Users/arunyahooda/Desktop/BT2O23/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"

# Where to save PNGs (these can be anywhere; they will be created if missing)
OUT_TRAIN  = "/Users/arunyahooda/Desktop/BT2O23/Training_png"
OUT_VAL    = "/Users/arunyahooda/Desktop/BT2O23/Validation_png"

# Which splits to process
DO_TRAIN = True
DO_VAL   = True

# ============ 2) SIMPLE OPTIONS ============

# Which anatomical axis to slice along:
#   "axial" (top→bottom), "coronal" (front→back), "sagittal" (left→right)
AXIS = "axial"

# Resize each PNG slice? Put (width, height) like (240, 240), or set to None to keep original size.
RESIZE_TO = None  # e.g., (240, 240) or None

# Save only slices that contain tumor (mask > 0)? Requires *_seg.nii* next to the volume.
ONLY_MASK_SLICES = False

# If a case has multiple modalities, we use this order for naming/recognition.
# (This does NOT filter modalities; we convert all non-seg volumes we find.)
MODALITY_PRIORITY = ["t1ce", "flair", "t2", "t1"]


# ============ 3) HELPER FUNCTIONS ============

def is_nifti(path):
    """True for .nii or .nii.gz"""
    base = path.lower()
    return base.endswith(".nii") or base.endswith(".nii.gz")

def is_seg_file(path):
    """True for *_seg.nii or *_seg.nii.gz (these are labels, not images)"""
    base = os.path.basename(path).lower()
    return base.endswith("_seg.nii") or base.endswith("_seg.nii.gz")

def stem_no_ext(path):
    """Filename without .nii/.nii.gz"""
    return re.sub(r"\.nii(\.gz)?$", "", os.path.basename(path), flags=re.IGNORECASE)

def case_id_from_stem(stem):
    """Remove trailing modality suffix (_t1ce/_flair/_t2/_t1) to get case id"""
    return re.sub(r"_(t1ce|flair|t2|t1)$", "", stem, flags=re.IGNORECASE)

def infer_modality(stem):
    """
    Try to read modality from filename end (t1ce/flair/t2/t1).
    If not found, return 'vol'.
    """
    s = stem.lower()
    for m in MODALITY_PRIORITY:
        if s.endswith(f"_{m}"):
            return m
    return "vol"

def find_mask_for(img_path):
    """
    Look for a matching segmentation mask next to the image file.
    We try the common names first, then any *_seg.nii* in the folder.
    """
    folder = os.path.dirname(img_path)
    stem = stem_no_ext(img_path)
    caseid = case_id_from_stem(stem)

    candidates = [
        os.path.join(folder, f"{caseid}_seg.nii.gz"),
        os.path.join(folder, f"{caseid}_seg.nii"),
        os.path.join(folder, f"{stem}_seg.nii.gz"),
        os.path.join(folder, f"{stem}_seg.nii"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    hits = glob.glob(os.path.join(folder, "*_seg.nii")) + \
           glob.glob(os.path.join(folder, "*_seg.nii.gz"))
    return hits[0] if hits else None

def load_volume_3d(path):
    """
    Load a NIfTI volume and reorient to RAS so axis order is consistent.
    Returns a 3D numpy array (float32).
    """
    img = nib.load(path)
    img = nib.as_closest_canonical(img)  # make orientation consistent
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume at {path}, got shape {vol.shape}")
    return vol

def robust_slice_to_u8(slice2d, p_lo=1, p_hi=99):
    """
    Normalize one 2D slice to uint8 (0..255) using robust percentiles.
    This avoids super dark/bright images caused by outliers.
    """
    arr = slice2d[np.isfinite(slice2d)]
    if arr.size == 0:
        arr = slice2d.ravel()
    lo, hi = np.percentile(arr, [p_lo, p_hi])
    x = np.clip(slice2d, lo, hi)
    x = x - x.min()
    m = x.max()
    if m > 0:
        x = x / m
    return (x * 255).astype(np.uint8)

def resize_if_needed(u8_img):
    """Resize grayscale mask or image to RESIZE_TO if set."""
    if RESIZE_TO is None:
        return u8_img
    w, h = RESIZE_TO  # (width, height)
    return cv2.resize(u8_img, (w, h), interpolation=cv2.INTER_LINEAR)

def resize_mask_if_needed(mask_u8):
    """
    Resize binary mask to RESIZE_TO using NEAREST so edges stay crisp.
    """
    if RESIZE_TO is None:
        return mask_u8
    w, h = RESIZE_TO
    return cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

def iter_axis_slices(vol):
    """
    Yield (index, 2D slice) along the chosen AXIS.
    - axial:    along Z (vol[:, :, k])
    - coronal:  along Y (vol[:, k, :])
    - sagittal: along X (vol[k, :, :])
    """
    if AXIS == "axial":
        for k in range(vol.shape[2]):
            yield k, vol[:, :, k]
    elif AXIS == "coronal":
        for k in range(vol.shape[1]):
            yield k, vol[:, k, :]
    elif AXIS == "sagittal":
        for k in range(vol.shape[0]):
            yield k, vol[k, :, :]
    else:
        raise ValueError("AXIS must be 'axial', 'coronal', or 'sagittal'")

def list_image_volumes(root_dir):
    """
    Recursively collect ALL NIfTI files under root_dir EXCEPT *_seg.nii*,
    which are labels (we use them only as masks).
    """
    files = []
    for r, _, fns in os.walk(root_dir):
        for fn in fns:
            p = os.path.join(r, fn)
            if is_nifti(p) and not is_seg_file(p):
                files.append(p)
    files.sort()
    return files


# ============ 4) CORE CONVERSION ============

def convert_one_volume(img_path, out_root):
    """
    Convert a single 3D NIfTI image into many 2D PNG slices.
    Also save a mask PNG per slice if a matching *_seg.nii* exists.
    """
    stem   = stem_no_ext(img_path)       # filename without .nii(.gz)
    caseid = case_id_from_stem(stem)     # e.g., "BraTS-GLI-00001-000"
    mod    = infer_modality(stem)        # e.g., "t1ce", "flair", ...

    # Load main volume
    vol = load_volume_3d(img_path)

    # Try to load mask (if present and same shape)
    mask = None
    mask_path = find_mask_for(img_path)
    if mask_path:
        try:
            m = load_volume_3d(mask_path)
            if m.shape == vol.shape:
                mask = (m > 0).astype(np.uint8)  # binary mask
            else:
                print(f"  (i) Mask shape mismatch for {os.path.basename(img_path)}: {m.shape} vs {vol.shape} (mask ignored).")
        except Exception as e:
            print(f"  (i) Mask load failed for {os.path.basename(img_path)}: {e}")

    # Create output folders
    img_dir  = os.path.join(out_root, caseid, mod, "images")
    mask_dir = os.path.join(out_root, caseid, mod, "masks") if mask is not None else None
    os.makedirs(img_dir, exist_ok=True)
    if mask_dir:
        os.makedirs(mask_dir, exist_ok=True)

    # Save each slice as PNG
    saved = 0
    for k, s2d in iter_axis_slices(vol):

        # If requested, skip slices with no tumor (requires mask)
        if mask is not None and ONLY_MASK_SLICES:
            if   AXIS == "axial":   m2d = mask[:, :, k]
            elif AXIS == "coronal": m2d = mask[:, k, :]
            else:                   m2d = mask[k, :, :]
            if m2d.max() == 0:
                continue

        # Normalize slice to 0..255 and (optionally) resize
        u8 = robust_slice_to_u8(s2d)
        u8 = resize_if_needed(u8)

        # Save image slice
        fname = f"{caseid}_{mod}_{AXIS}_{k:03d}.png"
        cv2.imwrite(os.path.join(img_dir, fname), u8)

        # Save mask slice (if exists)
        if mask_dir is not None:
            if   AXIS == "axial":   m2d = mask[:, :, k]
            elif AXIS == "coronal": m2d = mask[:, k, :]
            else:                   m2d = mask[k, :, :]
            m_u8 = (m2d > 0).astype(np.uint8) * 255  # 0 or 255
            m_u8 = resize_mask_if_needed(m_u8)
            cv2.imwrite(os.path.join(mask_dir, fname), m_u8)

        saved += 1

    print(f"  ✓ {os.path.basename(img_path)} -> {saved} PNGs at {os.path.join(out_root, caseid, mod)}")


def convert_folder(root_dir, out_dir, split_name=""):
    """
    Convert ALL non-seg NIfTI files under root_dir and save PNGs in out_dir.
    """
    if not os.path.isdir(root_dir):
        print(f"!! Skipping {split_name} (not a folder): {root_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    files = list_image_volumes(root_dir)
    if not files:
        print(f"!! No .nii/.nii.gz found under {root_dir}")
        return

    print(f"\n=== {split_name.upper()} ===")
    print(f"Found {len(files)} volume(s) in:\n  {root_dir}\nSaving to:\n  {out_dir}")
    print("Example files:")
    for p in files[:5]:
        print("  -", p)
    if len(files) > 5:
        print("  ...")

    done = 0
    for p in files:
        try:
            convert_one_volume(p, out_dir)
            done += 1
        except KeyboardInterrupt:
            print("\nUser interrupted. Stopping early.")
            break
        except Exception as e:
            print(f"  [x] Failed {os.path.basename(p)}: {e}")

    print(f"=== Finished {split_name.upper()}: {done}/{len(files)} converted ===")


# ============ 5) MAIN ============

def main():
    if DO_TRAIN:
        convert_folder(TRAIN_ROOT, OUT_TRAIN, split_name="train")
    if DO_VAL:
        convert_folder(VAL_ROOT,   OUT_VAL,   split_name="val")

    print("\nAll done!")
    if DO_TRAIN: print("  Train PNGs ->", OUT_TRAIN)
    if DO_VAL:   print("  Val   PNGs ->", OUT_VAL)


if __name__ == "__main__":
    main()
