# BUILD-NOTES — ham10000-shift-robust

## Dataset access (verified 2026-09-22)

- **DOI:** 10.7910/DVN/DBW86T (Harvard Dataverse, open, no credentialing).
- **License:** CC BY-NC 4.0 — non-commercial research use, attribution to
  Tschandl et al.
- File listing via Dataverse API:
  `GET https://dataverse.harvard.edu/api/datasets/%3ApersistentId/?persistentId=doi:10.7910/DVN/DBW86T`
  (note: the documented `:persistentId/versions/:latest-version/files` path
  returned `Illegal version identifier ':latest-version'`; the dataset-level
  endpoint with URL-encoded `%3ApersistentId` works.)
- Files (dataset version 4.0):
  | id | filename | size |
  |---|---|---|
  | 4338392 | HAM10000_metadata.tab (**tab-delimited, not .csv**) | 830,428 B |
  | 3172585 | HAM10000_images_part_1.zip | 1,366,522,108 B |
  | 3172584 | HAM10000_images_part_2.zip | 1,403,566,547 B |
  | 3838943 | HAM10000_segmentations_lesion_tschandl.zip | 10,808,743 B |
- Downloads: `https://dataverse.harvard.edu/api/access/datafile/<id>` ->
  303 redirect to a presigned S3 URL (`dvn-cloud-iqss.s3.amazonaws.com`, 1 h expiry).

## The `dataset` (acquisition source) column — confirmed from the real CSV

`vidir_molemax` 3954, `vidir_modern` 3363, `rosendahl` 2259, `vienna_dias` 439.
(No `vidir_old` in v4 metadata, contrary to some older writeups.) 10,015 rows,
7 dx classes: nv 6705, mel 1113, bkl 1099, bcc 514, akiec 327, vasc 142, df 115.

## Bugs / infra issues hit

1. **Full-zip download too slow** (~5 MB/min -> hours for 1.37 GB). Killed it and
   wrote `scripts/build_sample_v2.py`: parses the zip central directory from the
   file tail via HTTP Range requests (S3 supports `Range`, 206) and pulls only
   the needed JPEG entries (~520 files, ~141 MB total transfer instead of
   1.37 GB). Unreferenced JPEGs are pruned afterwards. The partial zip
   download was deleted; `data/` totals ~142 MB (< 150 MB budget).
2. **HEAD on the presigned S3 URL returns no Content-Length**; file size derived
   from `Range: bytes=-1` -> `Content-Range: bytes 0-0/<total>` instead.
3. **pip ran out of space**: `/tmp` is a 512 MB tmpfs and torch wheels are big.
   Fix: `export TMPDIR=~/.tmp-pip` before `pip install`.
4. `download.pytorch.org` read-timeouts during install — resolved with
   `--timeout 120 --retries 10` plus retries; CPU wheels installed fine.
5. Zip entries are stored (method 0) or deflated (method 8) — both handled.

## Honest numbers (incl. nulls)

Run 2026-09-22, 5 seeds (0–4), frozen ResNet-18 (ImageNet) + class-weighted
logistic probe, 192×192, n=520 sample. Full table: `results/run.json`.

- cross-source (train rosendahl → test vidir_modern): acc 0.257±0.014,
  macro-AUROC 0.708±0.012, macro-F1 0.224±0.019 (n train/val/test 164/54/167)
- pooled random split: acc 0.485±0.065, macro-AUROC 0.843±0.014,
  macro-F1 0.502±0.078 (n 364/78/78)
- **GAP pooled−cross-source: accuracy +0.227, macro-AUROC +0.136** ← the finding
- Triage (abstain if max-softmax < τ): cross-source acc on retained rises
  0.257 → 0.382 at τ=0.80 (85% referred); pooled 0.485 → 0.785 at τ=0.80
  (78% referred). Monotone, but gains need high referral — weak base model.
- **Null / negative results:**
  - First attempt (train vidir_molemax → test rosendahl, dx-stratified n=450)
    gave cross-source acc ≈ 0.08 — *worse than chance-looking* — because
    vidir_molemax has 0 akiec / 1 bcc in the whole population: pure label
    shift, not acquisition shift. Scrapped; protocol redesigned to
    rosendahl → vidir_modern (documented in README).
  - Per-source macro-AUROC on pooled test slices: NaN (undefined) — test
    slices of ~5–31 images miss classes entirely.
  - Absolute accuracies are low in both protocols; the claim is the gap,
    not SOTA performance.
