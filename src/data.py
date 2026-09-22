"""Data loading, sampling, and splitting for source-aware HAM10000 evaluation."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

METADATA_NAME = "HAM10000_metadata.tab"
ZIP_NAME = "HAM10000_images_part_1.zip"


def load_metadata(raw_dir=RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / METADATA_NAME, sep="\t")
    df = df[df["dx"].isin(CLASSES)].reset_index(drop=True)
    return df


def load_sample(root=None) -> pd.DataFrame:
    """Load the stratified sample produced by scripts/fetch_sample.py."""
    root = Path(root) if root else Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "data" / "processed" / "sample_manifest.csv")
    return df[df["dx"].isin(CLASSES)].reset_index(drop=True)


def sample_images(df: pd.DataFrame, n: int, seed: int = 0,
                  zip_path: Path | None = None) -> pd.DataFrame:
    """Stratified (by dx) sample of at most n images, restricted to those
    present in the downloaded zip. Writes extracted JPEGs to processed/."""
    zippath = Path(zip_path) if zip_path else RAW_DIR / ZIP_NAME
    with zipfile.ZipFile(zippath) as z:
        available = {Path(n_).stem for n_ in z.namelist()
                     if n_.lower().endswith(".jpg")}
    pool = df[df["image_id"].isin(available)].copy()
    rng = np.random.default_rng(seed)
    picked = []
    counts = (pool["dx"].value_counts() / len(pool) * n).round().astype(int)
    # ensure rare classes get at least a couple of examples
    counts = counts.clip(lower=2)
    for dx, k in counts.items():
        sub = pool[pool["dx"] == dx]
        k = min(k, len(sub))
        picked.append(sub.sample(k, random_state=int(rng.integers(1e9))))
    sample = pd.concat(picked).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    outdir = PROCESSED_DIR / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zippath) as z:
        namelist = {Path(n_).stem: n_ for n_ in z.namelist()}
        for img_id in sample["image_id"]:
            dest = outdir / f"{img_id}.jpg"
            if not dest.exists():
                dest.write_bytes(z.read(namelist[img_id]))
    return sample


def split_source_disjoint(sample: pd.DataFrame, train_source: str,
                          test_source: str, val_frac: float = 0.25,
                          seed: int = 0):
    """Train on all images from `train_source`, validate on a holdout of the
    same source, and test on `test_source` (different acquisition source)."""
    rng = np.random.default_rng(seed)
    src = sample[sample["dataset"] == train_source]
    idx = rng.permutation(len(src))
    n_val = max(1, int(val_frac * len(src)))
    return {
        "train": src.iloc[idx[n_val:]].reset_index(drop=True),
        "val": src.iloc[idx[:n_val]].reset_index(drop=True),
        "test": sample[sample["dataset"] == test_source].reset_index(drop=True),
    }


def split_pooled(sample: pd.DataFrame, train_frac: float = 0.7,
                 val_frac: float = 0.15, seed: int = 0):
    """Random stratified split (ignores source) — the usual, shift-hiding
    baseline. Implemented per-class so tiny classes (df=5) still land in
    every split instead of crashing sklearn's stratifier."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for dx, sub in sample.groupby("dx"):
        idx = rng.permutation(sub.index.to_numpy())
        n = len(idx)
        n_test = max(1, int(round((1 - train_frac - val_frac) * n)))
        n_val = max(1, int(round(val_frac * n)))
        n_test = min(n_test, n - 2)
        n_val = min(n_val, n - n_test - 1)
        parts["test"].append(sub.loc[idx[:n_test]])
        parts["val"].append(sub.loc[idx[n_test:n_test + n_val]])
        parts["train"].append(sub.loc[idx[n_test + n_val:]])
    return {k: pd.concat(v).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            for k, v in parts.items()}
