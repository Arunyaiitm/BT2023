# PURPOSE:
#   View ALL .nii / .nii.gz files inside a folder (and its subfolders)
#   as scrollable 2D slices. You can also overlay the tumor mask if a
#   matching "*_seg.nii(.gz)" file is present in the same case folder.


# CONTROLS:
#   - Scroll wheel or Up/Down arrows: move through slices
#   - Left/Right arrows: jump ±5 slices
#   - 'n' / 'p': next / previous file
#   - 'm': toggle mask overlay (if available)
#   - 'q': quit
# ------------------------------------------------------------

import os
import glob
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ====== EDIT THESE PATHS (put your folder paths here) ======
TRAIN_ROOT = "/Users/arunyahooda/Desktop/BT2O23/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
VAL_ROOT   = "/Users/arunyahooda/Desktop/BT2O23/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"

# Which folder to open right now? "train" or "val"
USE_ROOT   = "train"
# ===========================================================

# If a case has multiple modalities (t1ce, flair, t2, t1),
# we try to show the most useful one first.
MODALITY_PRIORITY = ["t1ce", "flair", "t2", "t1"]



# Helper functions (I/O)


def is_nifti(path: str) -> bool:
    """Return True if file is .nii or .nii.gz"""
    base = os.path.basename(path).lower()
    return base.endswith(".nii") or base.endswith(".nii.gz")

 
def is_seg_file(path: str) -> bool:
    """Return True if file looks like a segmentation mask: *_seg.nii*"""
    base = os.path.basename(path).lower()
    return base.endswith("_seg.nii") or base.endswith("_seg.nii.gz")


def load_nifti(path):
    """
    Load NIfTI and reorient to RAS (a common orientation).
    Returns: (3D volume as float32, voxel spacing)
    """
    img = nib.load(path)
    img = nib.as_closest_canonical(img)  # make axes consistent
    data = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]   # voxel spacing in mm
    return data, zooms


def normalize_to_uint8(slice2d, clip_percentiles=(1, 99)):
    """
    Convert a 2D slice to 0..255 uint8 using robust contrast.
    This makes the image look clearer (not too dark or too bright).
    """
    # Take only finite values to avoid NaN/inf issues
    arr = slice2d[np.isfinite(slice2d)]
    if arr.size == 0:
        arr = slice2d.flatten()
    lo, hi = np.percentile(arr, clip_percentiles)
    x = np.clip(slice2d, lo, hi)
    x = x - x.min()
    mx = x.max()
    if mx > 0:
        x = x / mx
    return (x * 255).astype(np.uint8)


def make_overlay_rgb(gray_u8, mask2d, alpha=0.35):
    """
    Put a red overlay where mask > 0, blended over the grayscale image.
    """
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.float32)
    m = (mask2d > 0)
    overlay = rgb.copy()
    overlay[m, 0] = 255  # red channel
    overlay[m, 1] = 0
    overlay[m, 2] = 0
    out = (1 - alpha) * rgb + alpha * overlay
    return out.astype(np.uint8)


def stem_without_ext(path: str) -> str:
    """Filename without the .nii / .nii.gz extension."""
    base = os.path.basename(path)
    return re.sub(r"\.nii(\.gz)?$", "", base, flags=re.IGNORECASE)


def find_mask_for_image(img_path: str):
    """
    Try to find a matching segmentation file for the given image.
    We first remove the modality suffix (like _t1ce) and then look for *_seg.nii*.
    If not found, we search any *_seg.nii* in the same folder.
    """
    folder = os.path.dirname(img_path)
    stem = stem_without_ext(img_path)
    # Remove trailing modality suffix to get a "case id"
    case_prefix = re.sub(r"_(t1ce|flair|t2|t1)$", "", stem, flags=re.IGNORECASE)

    # Most likely names
    candidates = [
        os.path.join(folder, f"{case_prefix}_seg.nii.gz"),
        os.path.join(folder, f"{case_prefix}_seg.nii"),
        os.path.join(folder, f"{stem}_seg.nii.gz"),
        os.path.join(folder, f"{stem}_seg.nii"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Fallback: any *_seg.nii* in this folder
    segs = glob.glob(os.path.join(folder, "*_seg.nii")) + \
           glob.glob(os.path.join(folder, "*_seg.nii.gz"))
    return segs[0] if segs else None


def collect_nifti_files(root_dir: str):
    """
    Find all .nii/.nii.gz files under root_dir (recursively),
    skip *_seg.nii* (we don't want to scroll those as images),
    and pick ONE modality per case folder using the priority list.
    """
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"Not a folder: {root_dir}")

    # Grab everything that looks like nifti
    all_files = sorted([p for p in glob.glob(os.path.join(root_dir, "**", "*.nii*"),
                                            recursive=True) if is_nifti(p)])

    # Group by case folder (BraTS keeps all modalities for a case in one folder)
    by_dir = {}
    for p in all_files:
        if is_seg_file(p):
            continue  # ignore masks here
        d = os.path.dirname(p)
        by_dir.setdefault(d, []).append(p)

    # For each case folder, choose the "best" modality by priority
    selected = []
    for _, files in by_dir.items():
        if len(files) == 1:
            selected.append(files[0])
        else:
            lower = [os.path.basename(x).lower() for x in files]
            picked = None
            for mod in MODALITY_PRIORITY:
                for i, name in enumerate(lower):
                    if name.endswith(f"_{mod}.nii") or name.endswith(f"_{mod}.nii.gz"):
                        picked = files[i]
                        break
                if picked:
                    break
            selected.append(picked or files[0])  # fallback if no match

    return sorted(selected)


# -------------------------
# The interactive viewer
# -------------------------

class FolderViewer:
    """
    Shows one 3D volume at a time.
    - Scroll through slices with mouse wheel or Up/Down.
    - Press 'n' / 'p' to move to next/previous file.
    - Press 'm' to toggle mask overlay (if available).
    """
    def __init__(self, file_list, root_name=""):
        if not file_list:
            raise FileNotFoundError("No .nii/.nii.gz files found in this folder.")
        self.files = file_list
        self.root_name = root_name

        # These will change as we load each file
        self.idx_file = 0
        self.vol = None
        self.mask = None
        self.zooms = None
        self.slice_idx = 0
        self.use_mask = True  # start with overlay ON (if mask exists)

        # Create the matplotlib window
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Load the first file and show it
        self.load_current_file()
        self.render()

        # Print controls in terminal
        print("\nControls:")
        print("  Scroll or ↑/↓ : previous/next slice")
        print("  ← / →         : jump ±5 slices")
        print("  n / p         : next / previous file")
        print("  m             : toggle mask overlay (if available)")
        print("  q             : quit\n")

        plt.show()

    def load_current_file(self):
        """Load the current file and (if possible) its matching mask."""
        path = self.files[self.idx_file]

        # Load volume
        self.vol, self.zooms = load_nifti(path)
        if self.vol.ndim != 3:
            raise ValueError(f"Expected a 3D volume at {path}, got shape {self.vol.shape}")

        # Start from the middle slice
        self.slice_idx = self.vol.shape[2] // 2

        # Try to load a mask
        self.mask = None
        mask_path = find_mask_for_image(path)
        if mask_path:
            try:
                m, _ = load_nifti(mask_path)
                if m.shape == self.vol.shape:
                    self.mask = (m > 0).astype(np.uint8)
                else:
                    print(f"(i) Mask shape mismatch for {os.path.basename(path)}: "
                          f"{m.shape} vs {self.vol.shape} — overlay disabled.")
            except Exception as e:
                print(f"(i) Could not load mask for {os.path.basename(path)}: {e}")

        # Update the window title and plot title
        title = (f"[{self.idx_file+1}/{len(self.files)}] {os.path.basename(path)} "
                 f"| {self.root_name} | shape {self.vol.shape} | spacing {self.zooms} mm")
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self.ax.set_title(title)

    def render(self):
        """Draw the current slice on screen (with/without mask)."""
        zmax = self.vol.shape[2] - 1
        self.slice_idx = max(0, min(self.slice_idx, zmax))

        # Take one 2D slice from the 3D volume
        s = self.vol[:, :, self.slice_idx]
        g = normalize_to_uint8(s)

        self.ax.clear()
        if self.use_mask and (self.mask is not None):
            m = self.mask[:, :, self.slice_idx]
            self.ax.imshow(make_overlay_rgb(g, m))
        else:
            self.ax.imshow(g, cmap="gray")
        self.ax.axis("off")

        # Update the title so you see which slice you're on
        base_title = self.ax.get_title().split("|")[0]
        self.ax.set_title(f"{base_title} | slice {self.slice_idx+1}/{zmax+1}")

        self.fig.canvas.draw_idle()

    # ------------- Event handlers -------------

    def on_scroll(self, event):
        """Mouse wheel: move one slice up/down."""
        if event.button == 'up':
            self.slice_idx += 1
        else:
            self.slice_idx -= 1
        self.render()

    def on_key(self, event):
        """Keyboard: arrows for slices, n/p for files, m for mask, q to quit."""
        if event.key in ['up']:
            self.slice_idx += 1
        elif event.key in ['down']:
            self.slice_idx -= 1
        elif event.key in ['right']:
            self.slice_idx += 5
        elif event.key in ['left']:
            self.slice_idx -= 5
        elif event.key == 'n':
            self.next_file()
        elif event.key == 'p':
            self.prev_file()
        elif event.key == 'm':
            self.use_mask = not self.use_mask
            self.render()
        elif event.key == 'q':
            plt.close(self.fig)
            return

        self.render()

    def next_file(self):
        """Go to next file in the list (wrap around at the end)."""
        self.idx_file = (self.idx_file + 1) % len(self.files)
        self.load_current_file()
        self.render()

    def prev_file(self):
        """Go to previous file in the list (wrap around at the start)."""
        self.idx_file = (self.idx_file - 1) % len(self.files)
        self.load_current_file()
        self.render()


# -------------------------
# Entry point (main)
# -------------------------

def main():
    # Pick which folder to open based on USE_ROOT
    root = TRAIN_ROOT if USE_ROOT.lower() == "train" else VAL_ROOT
    print(f"Using root: {root}")

    # Collect the files to view
    files = collect_nifti_files(root)
    print(f"Found {len(files)} image file(s). First few:")
    for f in files[:5]:
        print("  ", f)
    if len(files) > 5:
        print("  ...")

    # Start the viewer
    FolderViewer(files, root_name=USE_ROOT.upper())


if __name__ == "__main__":
    main()
