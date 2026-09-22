"""Build sample v2: joint (dataset x dx) stratification for source-shift study.

Why: vidir_molemax has ~zero akiec/bcc in the entire HAM10000 population, so
'train on molemax -> test on rosendahl' measures label shift, not acquisition
shift. v2 trains on rosendahl and tests on vidir_modern (both span all 7
classes in part_1), and keeps vidir_molemax / vienna_dias for extra per-source
reporting. Sampling caps per (dataset, dx) cell keep rare classes represented.

Reuses already-fetched JPEGs; only missing ones are range-fetched.
Writes: data/processed/sample_manifest.csv (v2), data/processed/images/*.jpg
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sample import (resolve_url, parse_central_directory, fetch_one,
                          OUT_IMG, OUT_MANIFEST)

ROOT = Path(__file__).resolve().parent.parent

# per-(dataset, dx) caps; min(pop, cap) taken
CAPS = {
    "rosendahl":    {"akiec": 40, "bcc": 40, "bkl": 40, "df": 15, "mel": 40,
                     "nv": 40, "vasc": 3},
    "vidir_modern": {"akiec": 8,  "bcc": 30, "bkl": 30, "df": 17, "mel": 30,
                     "nv": 30, "vasc": 22},
    "vidir_molemax": {"akiec": 0, "bcc": 1,  "bkl": 20, "df": 23, "mel": 13,
                      "nv": 20, "vasc": 20},
    "vienna_dias":  {"akiec": 0,  "bcc": 3,  "bkl": 4,  "df": 1,  "mel": 15,
                     "nv": 15, "vasc": 0},
}
SEED = 0
WORKERS = 8


def main():
    meta = pd.read_csv(ROOT / "data" / "raw" / "HAM10000_metadata.tab", sep="\t")
    part1 = set(json.load(open(ROOT / "data" / "raw" / "part1_ids.json")))
    pool = meta[meta["image_id"].isin(part1)].copy()
    print("part-1 population:", len(pool))

    rng = np.random.default_rng(SEED)
    picked = []
    for ds, caps in CAPS.items():
        for dx, cap in caps.items():
            sub = pool[(pool["dataset"] == ds) & (pool["dx"] == dx)]
            k = min(cap, len(sub))
            if k:
                picked.append(sub.sample(k, random_state=int(rng.integers(1e9))))
    sample = pd.concat(picked).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print("v2 sample:", len(sample))
    print(pd.crosstab(sample["dx"], sample["dataset"]))

    url = resolve_url()
    entries = parse_central_directory(url)
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    missing = [i for i in sample["image_id"]
               if not (OUT_IMG / f"{i}.jpg").exists()]
    print(f"already have {len(sample) - len(missing)}, fetching {len(missing)} ...")

    def job(img_id):
        for _ in range(4):
            try:
                return fetch_one(url, img_id, entries[img_id])
            except Exception as e:
                err = e
        return (img_id, str(err))

    ok, fails = 0, []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(job, missing):
            if isinstance(res, tuple):
                fails.append(res)
            else:
                ok += 1
    print(f"fetched ok={ok}, failed={len(fails)}")
    for f in fails[:10]:
        print("FAIL", f)
    sample.to_csv(OUT_MANIFEST, index=False)
    print("wrote", OUT_MANIFEST, f"({len(sample)} rows)")
    # prune JPEGs not referenced by the manifest (keep disk lean)
    keep = {f"{i}.jpg" for i in sample["image_id"]}
    pruned, kept_mb = 0, 0
    for p in OUT_IMG.glob("*.jpg"):
        if p.name not in keep:
            p.unlink()
            pruned += 1
        else:
            kept_mb += p.stat().st_size
    print(f"pruned {pruned} unreferenced images; kept {len(keep)} "
          f"({kept_mb / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
