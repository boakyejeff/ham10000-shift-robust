"""Fetch a stratified image sample from HAM10000_images_part_1.zip on Dataverse
using HTTP Range requests (the full zip is ~1.37 GB; we pull only ~400 JPEGs).

1. Resolves the Dataverse datafile URL -> presigned S3 URL.
2. Reads the zip central directory from the file tail (no full download).
3. Stratified-by-dx sample of image_ids present in part_1.
4. Range-fetches just those JPEG entries and writes them to data/processed/.

Usage: python scripts/fetch_sample.py --n 400 --seed 0
Writes: data/processed/sample_manifest.csv, data/processed/images/*.jpg
"""

import argparse
import io
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATAFILE_URL = "https://dataverse.harvard.edu/api/access/datafile/3172585"
METADATA = ROOT / "data" / "raw" / "HAM10000_metadata.tab"
OUT_IMG = ROOT / "data" / "processed" / "images"
OUT_MANIFEST = ROOT / "data" / "processed" / "sample_manifest.csv"

EOCD_SIG = b"\x50\x4b\x05\x06"
CDH_SIG = b"\x50\x4b\x01\x02"
LFH_SIG = b"\x50\x4b\x03\x04"


def resolve_url() -> str:
    r = requests.get(DATAFILE_URL, allow_redirects=False, timeout=30)
    assert r.status_code in (301, 302, 303, 307), r.status_code
    return r.headers["Location"]


def parse_central_directory(url: str) -> dict:
    """Return {stem: (local_header_offset, comp_size, method)} for .jpg entries."""
    # S3 does not return Content-Length on HEAD for this URL; use Content-Range
    r = requests.get(url, headers={"Range": "bytes=-1"}, timeout=30)
    assert r.status_code == 206, r.status_code
    total = int(r.headers["Content-Range"].split("/")[1])
    tail_n = min(total, 200_000)
    tail = requests.get(url, headers={"Range": f"bytes={total - tail_n}-{total - 1}"},
                        timeout=60).content
    eocd_pos = tail.rfind(EOCD_SIG)
    assert eocd_pos != -1, "EOCD not found in tail"
    (cd_count,) = struct.unpack("<H", tail[eocd_pos + 10:eocd_pos + 12])
    (cd_size,) = struct.unpack("<I", tail[eocd_pos + 12:eocd_pos + 16])
    (cd_offset,) = struct.unpack("<I", tail[eocd_pos + 16:eocd_pos + 20])
    entries = {}
    remaining, off = cd_size, cd_offset
    chunk = 4 << 20
    buf = b""
    while remaining > 0:
        n = min(chunk, remaining)
        buf += requests.get(url, headers={"Range": f"bytes={off}-{off + n - 1}"},
                            timeout=60).content
        off += n
        remaining -= n
    pos = 0
    for _ in range(cd_count):
        assert buf[pos:pos + 4] == CDH_SIG
        (method,) = struct.unpack("<H", buf[pos + 10:pos + 12])
        comp_size = struct.unpack("<I", buf[pos + 20:pos + 24])[0]
        name_len, extra_len, comment_len = struct.unpack("<HHH", buf[pos + 28:pos + 34])
        lho = struct.unpack("<I", buf[pos + 42:pos + 46])[0]
        name = buf[pos + 46:pos + 46 + name_len].decode("utf-8", "replace")
        pos += 46 + name_len + extra_len + comment_len
        if name.lower().endswith(".jpg"):
            entries[Path(name).stem] = (lho, comp_size, method)
    print(f"central directory: {cd_count} entries, {len(entries)} jpgs")
    return entries


def fetch_one(url, img_id, entry):
    lho, comp_size, method = entry
    hdr = requests.get(url, headers={"Range": f"bytes={lho}-{lho + 29}"},
                       timeout=60).content
    assert hdr[:4] == LFH_SIG, f"bad local header for {img_id}"
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    data_off = lho + 30 + name_len + extra_len
    data = requests.get(url, headers={"Range": f"bytes={data_off}-{data_off + comp_size - 1}"},
                        timeout=60).content
    assert len(data) == comp_size, f"short read {img_id}: {len(data)}/{comp_size}"
    if method == 8:
        data = zlib.decompress(data, -15)
    elif method != 0:
        raise ValueError(f"unsupported method {method} for {img_id}")
    (OUT_IMG / f"{img_id}.jpg").write_bytes(data)
    return img_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    meta = pd.read_csv(METADATA, sep="\t")
    url = resolve_url()
    entries = parse_central_directory(url)

    pool = meta[meta["image_id"].isin(entries)].copy()
    print(f"metadata rows: {len(meta)}, present in part_1: {len(pool)}")

    rng = np.random.default_rng(args.seed)
    counts = (pool["dx"].value_counts(normalize=True) * args.n).round().astype(int).clip(lower=2)
    picked = []
    for dx, k in counts.items():
        sub = pool[pool["dx"] == dx]
        picked.append(sub.sample(min(k, len(sub)),
                                 random_state=int(rng.integers(1e9))))
    sample = pd.concat(picked).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print("sample class counts:\n", sample["dx"].value_counts())
    print("sample source counts:\n", sample["dataset"].value_counts())

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    missing = [i for i in sample["image_id"]
               if not (OUT_IMG / f"{i}.jpg").exists()]
    print(f"fetching {len(missing)} images ...")
    ok, fails = 0, []
    def job(img_id):
        for attempt in range(4):
            try:
                return fetch_one(url, img_id, entries[img_id])
            except Exception as e:
                if attempt == 3:
                    return (img_id, str(e))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
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


if __name__ == "__main__":
    sys.exit(main())
