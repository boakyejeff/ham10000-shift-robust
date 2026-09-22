"""End-to-end run: sample -> features -> two protocols -> metrics -> triage.

Prereq: python scripts/fetch_sample.py  (writes data/processed/sample_manifest.csv
and data/processed/images/*.jpg via HTTP Range requests to Dataverse/S3).

Usage:
    python -m src.run --train-source vidir_molemax --test-source rosendahl \
        --seeds 0 1 2 3 4 --out results/run1.json

Per-seed variation comes from the split seeds and the probe seed (the sample
is fixed, features are deterministic given the sample).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import (load_sample, split_source_disjoint, split_pooled)
from .model import get_extractor, featurize, train_probe, predict_proba, encode
from .evaluate import metrics_for, per_source_metrics, triage_curve

ROOT = Path(__file__).resolve().parent.parent


def featurize_once(sample, extractor, preprocess):
    """Featurize every image once; index per split later (deterministic)."""
    feats = featurize(sample, extractor, preprocess)
    return feats


def run_protocol(sample, feats_all, split, seed):
    pos = {img: i for i, img in enumerate(sample["image_id"])}
    feats, y = {}, {}
    for k, df in split.items():
        idx = np.array([pos[i] for i in df["image_id"]])
        feats[k] = feats_all[idx]
        y[k] = encode(df)
    clf, scaler = train_probe(feats["train"], y["train"],
                              feats["val"], y["val"], seed=seed)
    proba = predict_proba(clf, scaler, feats["test"])
    res = metrics_for(y["test"], proba)
    res["per_class_support"] = {
        c: int((y["test"] == i).sum())
        for i, c in enumerate(
            ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])}
    test = split["test"].copy()
    test["y"] = y["test"]
    res["per_source"] = per_source_metrics(test, proba)
    res["triage"] = triage_curve(proba, y["test"])
    res["n_train"], res["n_val"], res["n_test"] = (
        len(split["train"]), len(split["val"]), len(split["test"]))
    res["train_sources"] = sorted(split["train"]["dataset"].unique().tolist())
    res["test_sources"] = sorted(split["test"]["dataset"].unique().tolist())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-source", default="rosendahl")
    ap.add_argument("--test-source", default="vidir_modern")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/run.json")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--img-size", type=int, default=192)
    args = ap.parse_args()

    sample = load_sample().reset_index(drop=True)
    print(f"sample: {len(sample)} images")
    print(sample["dx"].value_counts().to_string())
    print(sample["dataset"].value_counts().to_string())

    extractor, preprocess = get_extractor(args.arch, args.img_size)
    feats_all = featurize_once(sample, extractor, preprocess)
    print("features:", feats_all.shape)

    allres = {}
    for seed in args.seeds:
        disjoint = split_source_disjoint(sample, args.train_source,
                                         args.test_source, seed=seed)
        pooled = split_pooled(sample, seed=seed)
        r_dis = run_protocol(sample, feats_all, disjoint, seed)
        r_pool = run_protocol(sample, feats_all, pooled, seed)
        r_dis["protocol"] = f"train:{args.train_source}->test:{args.test_source}"
        r_pool["protocol"] = "pooled-random-split"
        allres[f"seed_{seed}"] = {"cross_source": r_dis, "pooled": r_pool}

    def agg(key):
        vals = {"cross_source": [], "pooled": []}
        for s in allres.values():
            for k in vals:
                v = s[k][key]
                if not (isinstance(v, float) and np.isnan(v)):
                    vals[k].append(v)
        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in vals.items()}

    summary = {
        "config": vars(args),
        "seeds": allres,
        "mean_std": {m: agg(m) for m in ["accuracy", "macro_auroc", "macro_f1"]},
    }
    g = {}
    for m in ["accuracy", "macro_auroc"]:
        g[m] = summary["mean_std"][m]["pooled"][0] - \
            summary["mean_std"][m]["cross_source"][0]
    summary["gap_pooled_minus_cross_source"] = g

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["mean_std"], indent=2))
    print("GAP (pooled - cross_source):", json.dumps(g, indent=2))
    print("wrote", outp)


if __name__ == "__main__":
    main()
