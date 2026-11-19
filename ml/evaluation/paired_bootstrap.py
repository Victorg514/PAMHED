"""
Paired bootstrap: is "text + features" better than "text only" on the SAME test users?

  python ml/evaluation/paired_bootstrap.py              # seed 42, Twitter test + eRisk where available
  python ml/evaluation/paired_bootstrap.py --seed 43

Each bootstrap sample draws test users with replacement and scores BOTH models on that same sample,
so user-to-user variation cancels out of the difference. This is much more sensitive than checking
whether two separate confidence intervals overlap. It covers test-set sampling noise only; training
noise (seeds) is a separate question.

Thresholds for macro F1 are each model's own Twitter validation thresholds (metrics.json).
Writes results/paired_bootstrap_s{seed}.md.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from features import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "out" / "v2"
TASKS = ["mental_health", "depression", "anxiety", "bipolar"]
FEATURES = "length+sentiment+language"


def load_pair(task, seed, pred_file):
    names = [f"{task}__none__s{seed}", f"{task}__{FEATURES}__s{seed}"]
    if not all((RUNS_DIR / n / pred_file).exists() for n in names):
        return None
    preds = [pd.read_parquet(RUNS_DIR / n / pred_file, columns=["tweet_id", "user_id", "y", "p1"]) for n in names]
    # Both runs write predictions in the same row order (tweet_id is not unique: some tweets appear twice)
    a, b = preds
    assert len(a) == len(b) and (a["tweet_id"].values == b["tweet_id"].values).all() \
        and (a["y"].values == b["y"].values).all(), "the two runs scored different rows"
    df = a.rename(columns={"p1": "p1_text"}).assign(p1_feat=b["p1"].values)
    metrics = [json.loads((RUNS_DIR / n / "metrics.json").read_text()) for n in names]
    thresholds = {"text": metrics[0]["val_user"]["threshold"], "feat": metrics[1]["val_user"]["threshold"]}
    return df, thresholds


def scores(df, users_df, thresholds):
    """Metric -> (text only, + features) on one (resampled) set of tweets and users."""
    out = {"tweet ROC-AUC": (roc_auc_score(df["y"], df["p1_text"]), roc_auc_score(df["y"], df["p1_feat"])),
           "user ROC-AUC": (roc_auc_score(users_df["y"], users_df["p1_text"]), roc_auc_score(users_df["y"], users_df["p1_feat"]))}
    out["user macro F1"] = tuple(
        f1_score(users_df["y"], (users_df[f"p1_{m}"] >= thresholds[m]).astype(int), average="macro", zero_division=0)
        for m in ("text", "feat"))
    return out


def paired_bootstrap(df, thresholds, n_boot, seed):
    users_df = df.groupby("user_id").agg(y=("y", "first"), p1_text=("p1_text", "mean"), p1_feat=("p1_feat", "mean"))
    observed = scores(df, users_df, thresholds)

    # Resample users; tweets follow their user
    tweet_idx = df.groupby("user_id").indices
    users = users_df.index.to_numpy()
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in observed}
    for _ in range(n_boot):
        sample = rng.choice(users, size=len(users), replace=True)
        d = df.iloc[np.concatenate([tweet_idx[u] for u in sample])]
        u = users_df.loc[sample]
        if d["y"].nunique() < 2 or u["y"].nunique() < 2:
            continue
        for k, (a, b) in scores(d, u, thresholds).items():
            diffs[k].append(b - a)

    rows = []
    for k, (a, b) in observed.items():
        diff = np.array(diffs[k])
        p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))  # two-sided
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({"metric": k, "text only": a, "+ features": b, "difference": b - a, "ci": (lo, hi),
                     "p": p, "features better in": (diff > 0).mean()})
    return rows, users_df["y"].value_counts().to_dict()


def main(args):
    sections = [f"# Paired bootstrap: text + features vs text only (seed {args.seed})", "",
                f"{args.n_boot} bootstrap samples of test users, both models scored on the same sample. "
                "Difference = features minus text only. p = two-sided bootstrap p-value. Covers test-set "
                "sampling noise, not training noise (seeds).", ""]
    for label, pred_file, tasks in [("Twitter test set", "pred_test.parquet", TASKS),
                                    ("eRisk Reddit (depressed vs control users)", "pred_erisk.parquet", TASKS)]:
        body = []
        for task in tasks:
            pair = load_pair(task, args.seed, pred_file)
            if pair is None:
                continue
            rows, users = paired_bootstrap(*pair, args.n_boot, args.seed)
            print(f"\n{label} | {task} | users {users}")
            for r in rows:
                body.append(f"| {task} | {r['metric']} | {r['text only']:.3f} | {r['+ features']:.3f} | "
                            f"{r['difference']:+.3f} | [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['p']:.3f} | "
                            f"{r['features better in']:.0%} |")
                print(f"  {r['metric']:<14} {r['text only']:.4f} -> {r['+ features']:.4f}  diff {r['difference']:+.4f}  "
                      f"CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  p={r['p']:.3f}")
        if body:
            sections += [f"## {label}", "",
                         "| Task | Metric | Text only | + features | Difference | 95% CI | p | Features better in |",
                         "|---|---|---|---|---|---|---|---|"] + body + [""]
    out = PROJECT_ROOT / "results" / f"paired_bootstrap_s{args.seed}.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_boot", type=int, default=1000)
    main(p.parse_args())
