"""
Cross-platform test: Twitter-trained v2 models on eRisk Reddit users (depression vs control).

  python ml/preprocessing/process_erisk_v2.py    # once: builds data/erisk_v2.parquet
  python ml/evaluation/evaluate_erisk_v2.py      # writes results/erisk_cross_platform.md

Nothing is fit or tuned on eRisk:
- features are scaled with each model's Twitter train-split statistics (config.json),
- decision thresholds are the ones tuned on the Twitter validation split (metrics.json),
- all eRisk users are used (no subsampling); ROC-AUC does not depend on the class ratio.
eRisk positives are depressed users, so the mental-health screener is scored on the same labels.
Predictions are cached in out/v2/<run>/pred_erisk.parquet.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))
from features import PROJECT_ROOT, apply_scaler
from evaluation.metrics import binary_scores, to_user_level, bootstrap_ci
from train import FusionModel, TweetDataset, make_collate, predict

ERISK_FILE = PROJECT_ROOT / "data" / "erisk_v2.parquet"
RUNS_DIR = PROJECT_ROOT / "out" / "v2"
DEFAULT_RUNS = ["depression__none__s42", "depression__length+sentiment+language__s42",
                "mental_health__none__s42", "mental_health__length+sentiment+language__s42"]


def run_predictions(run_dir, df, device, batch):
    cache = run_dir / "pred_erisk.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    config = json.loads((run_dir / "config.json").read_text())
    x = apply_scaler(df, config)
    model = FusionModel(config["base_model"], config["num_labels"], len(config["features"]), config["dropout"]).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else None
    loader = DataLoader(TweetDataset(df["text"].tolist(), x, df["label"].to_numpy()), batch_size=batch,
                        collate_fn=make_collate(AutoTokenizer.from_pretrained(config["base_model"]), config["max_len"]))
    proba = predict(model, loader, device, amp_dtype)
    pred = pd.DataFrame({"tweet_id": df["tweet_id"], "user_id": df["user_id"], "y": df["label"].to_numpy(), "p1": proba[:, 1]})
    pred.to_parquet(cache, index=False)
    del model
    torch.cuda.empty_cache()
    return pred


def score(pred, metrics, n_boot):
    t_post, t_user = metrics["val_tweet"]["threshold"], metrics["val_user"]["threshold"]
    post = binary_scores(pred["y"].to_numpy(), pred["p1"].to_numpy(), t_post)
    user_df = to_user_level(pred, ["p1"])
    user = binary_scores(user_df["y"].to_numpy(), user_df["p1"].to_numpy(), t_user)
    if n_boot:
        post["roc_auc_ci"] = bootstrap_ci(pred, lambda d: roc_auc_score(d["y"], d["p1"]), n_boot)
        user["roc_auc_ci"] = bootstrap_ci(user_df, lambda d: roc_auc_score(d["y"], d["p1"]), n_boot)
        user["macro_f1_ci"] = bootstrap_ci(
            user_df, lambda d: f1_score(d["y"], (d["p1"] >= t_user).astype(int), average="macro", zero_division=0), n_boot)
    return {"post": post, "user": user}


def fmt(m, key):
    ci = m.get(f"{key}_ci")
    return f"{m[key]:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    needed = sorted({f for r in args.runs for f in json.loads((RUNS_DIR / r / "config.json").read_text())["features"]})
    df = pq.read_table(ERISK_FILE, columns=["tweet_id", "user_id", "label", "text"] + needed).to_pandas()
    users = df.drop_duplicates("user_id")["label"].value_counts().to_dict()
    print(f"eRisk: {len(df)} posts, users {users} (1 = depression)")

    results, rows = {}, []
    for run in args.runs:
        run_dir = RUNS_DIR / run
        print(f"\n=== {run}")
        pred = run_predictions(run_dir, df, device, args.batch)
        twitter = json.loads((run_dir / "metrics.json").read_text())
        res = score(pred, twitter, args.n_boot)
        results[run] = res
        print(f"post: ROC-AUC {res['post']['roc_auc']:.4f} macro F1 {res['post']['macro_f1']:.4f} | "
              f"user: ROC-AUC {res['user']['roc_auc']:.4f} macro F1 {res['user']['macro_f1']:.4f}")
        task, feats, _ = run.split("__")
        rows.append([task, "text only" if feats == "none" else feats.replace("+", ", "),
                     f"{twitter['test_user']['roc_auc']:.3f}", fmt(res["post"], "roc_auc"), fmt(res["user"], "roc_auc"),
                     fmt(res["user"], "macro_f1"), f"{res['user']['pos_recall']:.2f}", f"{res['user']['pred_pos_rate']:.2f}"])

    out = PROJECT_ROOT / "results" / "erisk_cross_platform.md"
    head = ["Model trained for", "Features", "Twitter test user AUC", "eRisk post AUC", "eRisk user AUC",
            "eRisk user macro F1", "User recall", "Users flagged"]
    lines = [
        "# Cross-platform test: Twitter-trained models on eRisk (Reddit)", "",
        f"eRisk depression collection, users with >= 50 posts: {users.get(1, 0)} depressed, {users.get(0, 0)} control "
        f"({len(df)} post-baseline posts). Positive = depressed user. No retraining; thresholds tuned on the Twitter "
        "validation split; 95% intervals resample users.", "",
        "| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (PROJECT_ROOT / "results" / "erisk_cross_platform.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n_boot", type=int, default=1000)
    main(p.parse_args())
