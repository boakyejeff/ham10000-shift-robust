# Source-Aware Evaluation of Skin Lesion Classification on HAM10000

## Novelty claim

Most HAM10000 tutorials do **random train/test splits** and report ~85-90% accuracy.
That number is misleading: HAM10000 pools images from **four different acquisition
sources** (dermoscope hardware, sites, eras), and a random split puts images from
*every* source in both train and test. The model is never tested on truly unseen
acquisition conditions.

This repo evaluates the way a deployed model would actually face the world:

- **Cross-source protocol:** train on one acquisition source (`rosendahl`, a
  Danish site), test on a *different* source (`vidir_modern`, Vienna) —
  fully source-disjoint.
- **Pooled baseline:** ordinary random stratified split (the usual demo), for comparison.
- **The finding is the GAP** between the two: how much performance a random split
  overstates because it leaks acquisition conditions into the test set.

> Design note: the largest source, `vidir_molemax`, contains **zero** `akiec`
> and one `bcc` image in the entire dataset (see `BUILD-NOTES.md`), so training
> on it would measure label shift, not acquisition shift. `rosendahl` →
> `vidir_modern` is the largest source pair spanning all 7 classes.

Plus an uncertainty-gated **triage curve**: refer low-confidence cases to a
dermatologist (abstain when max-softmax < τ) and see accuracy-vs-referral-rate.

## Model

CPU-feasible by design: a **frozen ImageNet-pretrained ResNet-18** (torchvision)
as a feature extractor (192×192 input → 512-d features) plus a **class-weighted
logistic regression** linear probe. With only ~520 images, training a CNN from
scratch would memorize noise; fine-tuning on 2 CPU cores is slow; the frozen
backbone + linear probe keeps the experiment about the *evaluation protocol*,
not about squeezing accuracy. Logistic-regression `C` is tuned on a validation
split (0.01–10); class weights handle the heavy `nv`-majority imbalance.

## How to run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # torch CPU, torchvision, scikit-learn, pandas, pillow, numpy

# 1. Fetch the joint (source x class) stratified image sample (~520 JPEGs) via
#    HTTP Range requests from the public Dataverse DOI 10.7910/DVN/DBW86T
#    (no credentialing needed; avoids the 1.37 GB zip download)
python scripts/build_sample_v2.py

# 2. Run both protocols (5 seeds) -> results/run.json
python -m src.run --train-source rosendahl --test-source vidir_modern \
    --seeds 0 1 2 3 4 --out results/run.json
```

Smoke test: `python -m src.run --seeds 0 --out results/smoke.json` (reuses the
sample; ~1–2 min on CPU).

## Results

> Real numbers from real runs on this machine (5 seeds, frozen ResNet-18 +
> logistic probe, 192×192, n=520). Sample is small so CIs are wide —
> see Limitations.

| protocol | test source(s) | n train / val / test | accuracy | macro-AUROC | macro-F1 |
|---|---|---|---|---|---|
| cross-source (train `rosendahl` → test `vidir_modern`) | vidir_modern only | 164 / 54 / 167 | **0.257 ± 0.014** | **0.708 ± 0.012** | 0.224 ± 0.019 |
| pooled random split | all 4 sources | 364 / 78 / 78 | **0.485 ± 0.065** | **0.843 ± 0.014** | 0.502 ± 0.078 |
| **GAP (pooled − cross-source)** | — | — | **+0.227** | **+0.136** | +0.277 |

**Reading:** absolute accuracy is modest in both protocols (520 images, 7 classes,
linear probe — this is not a SOTA claim). The finding is the *gap*: the ordinary
random-split evaluation overstates accuracy by ~23 points and macro-AUROC by
~0.14 relative to testing on a truly unseen acquisition source. Random splits
leak every site's imaging conditions into the test set; source-disjoint
evaluation removes that leak.

## Per-source breakdown

<!--PERSOURCE-->

| protocol | test source | n (test) | accuracy (mean ± std, 5 seeds) |
|---|---|---|---|
| cross-source | vidir_modern | 167 | 0.257 ± 0.014 |
| pooled | rosendahl | ~31 | 0.435 ± 0.171 |
| pooled | vidir_modern | ~25 | 0.430 ± 0.060 |
| pooled | vidir_molemax | ~16 | 0.630 ± 0.113 |
| pooled | vienna_dias | ~5 | 0.446 ± 0.265 |

(Per-source macro-AUROC is undefined on these tiny per-source test slices —
too few classes present — so only accuracy is reported here. The ± stds are
large because each pooled test slice holds a handful of images; that noise is
the honest price of a 520-image sample.)

## Triage curve (mean over 5 seeds)

Uncertainty-gated triage: abstain (refer to dermatologist) when max-softmax < τ.

| τ | cross-source: referral rate | cross-source: acc on retained | pooled: referral rate | pooled: acc on retained |
|---|---|---|---|---|
| 0.00 | 0.000 | 0.257 | 0.000 | 0.485 |
| 0.20 | 0.001 | 0.258 | 0.000 | 0.485 |
| 0.40 | 0.255 | 0.267 | 0.233 | 0.539 |
| 0.60 | 0.640 | 0.319 | 0.538 | 0.657 |
| 0.80 | 0.847 | 0.382 | 0.779 | 0.785 |

Referring low-confidence cases monotonically improves accuracy on the retained
cases in both protocols — but the model is weak enough that large gains demand
high referral rates (e.g. +12 pts cross-source at 85% referral). Triage helps;
it does not fix a weak base model.

## Limitations

- **Small sample (520 of 10,015 images):** class counts per source are small
  (e.g. 3 `vasc` in rosendahl, 8 `akiec` in vidir_modern), so per-class numbers
  are noisy. Reported means ± std over 5 seeds; per-source AUROC is undefined
  when classes are absent from a tiny test slice.
- **Single backbone / single probe:** frozen ResNet-18 + logistic regression is a
  deliberate cheap choice, not a SOTA claim.
- **No clinical use:** research demo only; not a diagnostic tool.
- **License:** HAM10000 is CC BY-NC 4.0 (Tschandl et al.) — non-commercial
  research use with attribution.

## Layout

```
src/
  data.py      metadata loading, stratified sampling, source-disjoint & pooled splits
  model.py     frozen torchvision extractor + class-weighted logistic probe
  evaluate.py  accuracy / macro-AUROC / macro-F1, per-source metrics, triage curve
  run.py       orchestrates both protocols over seeds, writes results/run.json
scripts/
  fetch_sample.py   range-request sample fetch from Dataverse (no 1.4 GB download)
BUILD-NOTES.md      exact URLs, sizes, bugs hit, honest numbers incl. nulls
```
