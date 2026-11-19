"""
CPU feature-group ablation on the behavioural features alone (no text), to decide which groups are
worth the GPU time.

  python ml/analysis/feature_ablation.py --task mental_health

For each feature set it fits a gradient-boosted tree model on train and reports validation ROC-AUC,
per tweet and per user (mean probability over the user's tweets). Uses the VALIDATION split only,
so the test split stays untouched for the paper.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from features import (PROJECT_ROOT, FEATURE_GROUPS, ALL_FEATURES, resolve_features, task_labels, load_split)


def feature_sets():
    sets = {"all (incl. redundant)": ALL_FEATURES, "default": resolve_features("default"),
            "length+sentiment+language": resolve_features("length,sentiment,language")}
    for g in FEATURE_GROUPS:
        sets[f"only {g}"] = resolve_features(g)
    for f in resolve_features("default"):
        sets[f"single: {f}"] = [f]
    for g in FEATURE_GROUPS:
        sets[f"default minus {g}"] = resolve_features(f"default,-{g}")
    return sets


def score(train, y_train, val, y_val, feats, seed):
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True,
                                           validation_fraction=0.1, class_weight="balanced", random_state=seed)
    model.fit(train[feats].to_numpy(), y_train)
    p = model.predict_proba(val[feats].to_numpy())[:, 1]
    user = pd.DataFrame({"user_id": val["user_id"], "p": p, "y": y_val}).groupby("user_id").agg(p=("p", "mean"), y=("y", "first"))
    return roc_auc_score(y_val, p), roc_auc_score(user["y"], user["p"])


def main(args):
    train = load_split("train", ALL_FEATURES, with_text=False)
    val = load_split("val", ALL_FEATURES, with_text=False)
    if args.train_sample and len(train) > args.train_sample:
        train = train.sample(n=args.train_sample, random_state=args.seed)
    y_train, y_val = task_labels(args.task, train["label"]), task_labels(args.task, val["label"])
    print(f"Task {args.task}: train={len(train)} val={len(val)} rows, val users={val['user_id'].nunique()}")

    rows = []
    for name, feats in feature_sets().items():
        t0 = time.time()
        tweet_auc, user_auc = score(train, y_train, val, y_val, feats, args.seed)
        rows.append({"feature set": name, "n": len(feats), "val tweet ROC-AUC": tweet_auc, "val user ROC-AUC": user_auc})
        print(f"{name:<28} n={len(feats):>2}  tweet AUC {tweet_auc:.4f}  user AUC {user_auc:.4f}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / f"feature_ablation_{args.task}.md"
    out.parent.mkdir(exist_ok=True)
    header = (f"# Feature ablation: {args.task}\n\nBehavioural features only (no text), gradient-boosted trees, "
              f"validation split. 0.5 = chance.\n\n")
    table = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    table += ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    out.write_text(header + "\n".join(table) + "\n", encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["mental_health", "depression", "anxiety", "bipolar"])
    p.add_argument("--train_sample", type=int, default=200_000, help="Subsample train rows for speed (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
