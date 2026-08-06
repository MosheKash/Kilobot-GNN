"""
quickdraw_to_png.py
====================
Downloads QuickDraw .npy bitmap files from Google Cloud Storage,
merges all categories into a single flat folder of 64×64 PNGs.
Designed as a formation-image pool for swarm RL training — no labels,
no subfolders, just a large directory you can sample from at random.

Output layout
-------------
output_dir/
    000000.png
    000001.png
    ...
    N total images, shuffled across all categories

Requirements
------------
    pip install numpy pillow tqdm

Usage examples
--------------
    # 5 categories, up to 1 000 images each, binarized (default):
    python quickdraw_to_png.py --categories cat dog bicycle airplane tree \
        --max-per-class 1000 --output-dir ./formations

    # Cap the whole pool at 50 000 images regardless of per-class count:
    python quickdraw_to_png.py --categories cat dog bicycle airplane tree \
        --max-total 50000 --output-dir ./formations

    # Use .npy files you already downloaded locally:
    python quickdraw_to_png.py --npy-dir ./my_npy_files \
        --max-per-class 2000 --output-dir ./formations

    # Keep grayscale instead of binarizing:
    python quickdraw_to_png.py --categories cat dog \
        --max-per-class 1000 --no-binarize --output-dir ./formations

    # All 345 categories, 500 images each (~170 K images total):
    python quickdraw_to_png.py --all-categories \
        --max-per-class 500 --output-dir ./formations

Loading images during training (PyTorch example)
-------------------------------------------------
    import os, random
    from PIL import Image
    import numpy as np

    pool = [f for f in os.listdir("./formations") if f.endswith(".png")]

    def load_random_formation(folder="./formations"):
        path = os.path.join(folder, random.choice(pool))
        return np.array(Image.open(path))   # shape (64, 64), dtype uint8
"""

import argparse
import os
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── GCS / repo constants ──────────────────────────────────────────────────────
GCS_HTTPS    = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
CATEGORIES_URL = (
    "https://raw.githubusercontent.com/googlecreativelab/"
    "quickdraw-dataset/master/categories.txt"
)
SOURCE_SIZE  = 28
TARGET_SIZE  = 28
DEFAULT_THRESHOLD = 128   # pixels ≥ this value become 255 (white stroke) when binarizing

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_all_categories() -> list[str]:
    print("Fetching category list …")
    with urllib.request.urlopen(CATEGORIES_URL) as r:
        return [l.strip() for l in r.read().decode().splitlines() if l.strip()]


def download_npy(category: str, cache_dir: Path) -> Path:
    safe = category.replace(" ", "_")
    dest = cache_dir / f"{safe}.npy"
    if dest.exists():
        return dest
    url = f"{GCS_HTTPS}/{category.replace(' ', '%20')}.npy"
    try:
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024,
                  miniters=1, desc=f"  {category}", leave=False) as bar:
            def _hook(count, block_size, total):
                if total > 0:
                    bar.total = total
                bar.update(count * block_size - bar.n)
            urllib.request.urlretrieve(url, dest, reporthook=_hook)
    except Exception as exc:
        dest.unlink(missing_ok=True)   # remove partial file on failure
        raise RuntimeError(f"Could not download '{category}': {exc}") from exc
    return dest


def load_npy(path: Path, max_samples: int | None, rng: np.random.Generator) -> np.ndarray:
    """Load .npy, return up to max_samples randomly chosen rows."""
    data = np.load(path)                         # (N, 784) uint8
    if max_samples is not None and len(data) > max_samples:
        idx  = rng.choice(len(data), size=max_samples, replace=False)
        data = data[idx]
    return data


def to_png(flat: np.ndarray, out_path: Path, binarize: bool, threshold: int) -> None:
    """
    flat: 1-D uint8 array of length 784 (28×28).
    Pixel value 255 = stroke, 0 = background in the raw QuickDraw data.
    We keep that convention (white stroke on black background matches
    a lit-pixel / occupied-cell view of the formation).
    Resize to TARGET_SIZE × TARGET_SIZE, optionally binarize, save PNG.
    """
    img = Image.fromarray(flat.reshape(SOURCE_SIZE, SOURCE_SIZE), mode="L")
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    if binarize:
        img = img.point(lambda p: 255 if p >= threshold else 0)
    img.save(out_path, format="PNG")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge QuickDraw .npy files into a flat pool of 64×64 PNGs."
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--categories", nargs="+", metavar="NAME",
                     help="Category names to download (e.g. cat dog tree). "
                          "Omit to use all 345 categories.")
    src.add_argument("--npy-dir", type=Path, metavar="DIR",
                     help="Local directory of already-downloaded .npy files.")

    parser.add_argument("--output-dir",    type=Path,  default=Path("./formations"),
                        help="Output directory for PNGs (default: ./formations).")
    parser.add_argument("--npy-cache",     type=Path,  default=Path("./npy_cache"),
                        help="Cache dir for downloaded .npy files (default: ./npy_cache).")
    parser.add_argument("--max-per-class", type=int,   default=None,  metavar="N",
                        help="Max images sampled per category. Default: all.")
    parser.add_argument("--max-total",     type=int,   default=None,  metavar="N",
                        help="Hard cap on total images written (random subset of merged pool).")
    parser.add_argument("--no-binarize",   action="store_true",
                        help="Keep grayscale instead of thresholding to black/white.")
    parser.add_argument("--threshold",     type=int,   default=DEFAULT_THRESHOLD,
                        help=f"Binarization threshold 0-255 (default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed (default: 42).")

    args   = parser.parse_args()
    binarize = not args.no_binarize
    rng    = np.random.default_rng(args.seed)

    # ── Resolve .npy paths ────────────────────────────────────────────────────
    if args.npy_dir:
        npy_paths  = sorted(args.npy_dir.glob("*.npy"))
        if not npy_paths:
            sys.exit(f"No .npy files found in {args.npy_dir}")
        categories = [p.stem.replace("_", " ") for p in npy_paths]
        print(f"Found {len(npy_paths)} .npy files in {args.npy_dir}")
    else:
        if args.categories:
            categories = args.categories
        else:
            # nothing specified (or --categories omitted) → use all 345
            categories = fetch_all_categories()

        args.npy_cache.mkdir(parents=True, exist_ok=True)
        cached    = sum(1 for c in categories if (args.npy_cache / f"{c.replace(' ','_')}.npy").exists())
        to_fetch  = len(categories) - cached
        print(f"Downloading {len(categories)} categor(y/ies) "
              f"({cached} already cached, {to_fetch} to fetch) …")
        npy_paths, valid_cats = [], []
        with tqdm(total=len(categories), unit="cat", desc="Categories") as outer:
            for cat in categories:
                try:
                    npy_paths.append(download_npy(cat, args.npy_cache))
                    valid_cats.append(cat)
                except RuntimeError as e:
                    tqdm.write(f"  WARNING: {e} — skipping.")
                finally:
                    outer.update(1)
        categories = valid_cats

    # ── Collect all flat arrays into memory-mapped list ───────────────────────
    # For very large runs we stream directly to disk instead of RAM.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting to {'binary' if binarize else 'grayscale'} "
          f"{TARGET_SIZE}×{TARGET_SIZE} PNGs …")

    # Two-pass approach: first count so we can pre-shuffle the global index,
    # then write. For huge datasets we just write sequentially and shuffle filenames after.
    all_arrays: list[np.ndarray] = []

    for cat, npy_path in zip(categories, npy_paths):
        data = load_npy(npy_path, args.max_per_class, rng)
        all_arrays.append(data)
        print(f"  {cat}: {len(data):,} images loaded")

    merged = np.concatenate(all_arrays, axis=0)   # (Total, 784)
    del all_arrays

    # Optional global cap
    if args.max_total is not None and len(merged) > args.max_total:
        idx    = rng.choice(len(merged), size=args.max_total, replace=False)
        merged = merged[idx]
        print(f"  Capped to {args.max_total:,} images (--max-total)")

    # Shuffle so the on-disk ordering is random (helps if training reads sequentially)
    shuffle_idx = np.arange(len(merged))
    rng.shuffle(shuffle_idx)
    merged = merged[shuffle_idx]

    n_digits = len(str(len(merged) - 1))
    print(f"\nWriting {len(merged):,} PNGs to {args.output_dir}/ …")

    for i, flat in enumerate(tqdm(merged, unit="img")):
        fname = f"{i:0{n_digits}d}.png"
        to_png(flat, args.output_dir / fname, binarize, args.threshold)

    print("\n" + "═" * 50)
    print(f"Done!  {len(merged):,} PNGs in {args.output_dir}/")
    print(f"  Size : {TARGET_SIZE}×{TARGET_SIZE} px")
    print(f"  Mode : {'binary (black/white)' if binarize else 'grayscale'}")
    print(f"  Load : random.choice(os.listdir('{args.output_dir}'))")
    print("═" * 50)


if __name__ == "__main__":
    main()
