"""
SHAP for a v2 fusion model, as a script (same logic as notebooks/SHAP_v2.ipynb, but writes files).

  python ml/analysis/shap_analysis.py                       # mental health + features, seed 42
  python ml/analysis/shap_analysis.py --run depression__length+sentiment+language__s42

Outputs (results/):
  figures/shap_features_<task>.png   beeswarm of the behavioural features, text held fixed
  shap_text_<task>.html              word-level explanations for 12 sampled tweets, features held fixed
  shap_<task>.md                     mean |SHAP| per feature, gate statistics, and the sampled tweets

The two inputs are explained separately, each with the other held fixed. Examples are sampled at random
from the test split, from both classes and from correct and wrong predictions (at most one per user).
"""
import argparse
import json
import pathlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml")); sys.path.insert(0, str(ROOT / "ml" / "training"))
from features import load_split, apply_scaler, task_labels  # noqa: E402
from train import FusionModel  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main(args):
    run_dir = ROOT / "out" / "v2" / args.run
    cfg = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    features, task = cfg["features"], cfg["task"]
    threshold = metrics["test_tweet"]["threshold"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    model = FusionModel(cfg["base_model"], cfg["num_labels"], len(features), cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device)); model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])

    def predict_proba(texts, feats, batch_size=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(list(texts[i:i + batch_size]), truncation=True, max_length=cfg["max_len"],
                                padding=True, return_tensors="pt").to(device)
                delta = torch.as_tensor(feats[i:i + batch_size], dtype=torch.float32, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(enc["input_ids"], enc["attention_mask"], delta)
                out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    # --- sample: TP / TN / FP / FN, one tweet per user at most ---
    test_df = load_split("test", features)
    test_df["y"] = task_labels(task, test_df["label"].to_numpy())
    pred = pd.read_parquet(run_dir / "pred_test.parquet")[["tweet_id", "p1"]]
    assert pred["tweet_id"].is_unique
    test_df = test_df.merge(pred, on="tweet_id", how="left")
    test_df["pred"] = (test_df["p1"] >= threshold).astype(int)
    test_df["outcome"] = np.select(
        [(test_df.y == 1) & (test_df.pred == 1), (test_df.y == 0) & (test_df.pred == 0), (test_df.y == 0) & (test_df.pred == 1)],
        ["TP", "TN", "FP"], "FN")
    sample = (test_df.sample(frac=1, random_state=args.seed).drop_duplicates("user_id")
              .groupby("outcome", group_keys=False).apply(lambda g: g.head(args.n_examples // 4))).reset_index(drop=True)
    print("sample:", sample["outcome"].value_counts().to_dict())

    lines = [f"# SHAP: `{args.run}`", "",
             f"Task {task}, {len(features)} features, test threshold {threshold:.3f}. {len(sample)} test tweets sampled "
             f"(at most one per user): " + ", ".join(f"{k} {v}" for k, v in sample["outcome"].value_counts().items()) + ".", ""]

    # --- 1. behavioural features, text fixed ---
    train_bg = load_split("train", features, with_text=False)
    background = shap.kmeans(apply_scaler(train_bg.sample(5000, random_state=args.seed), cfg), 20)
    feat_shap = []
    for i, row in sample.iterrows():
        x = apply_scaler(row.to_frame().T, cfg)
        f = lambda F, text=row["text"]: predict_proba([text] * len(F), F)
        feat_shap.append(shap.KernelExplainer(f, background).shap_values(x, nsamples=200, silent=True)[0])
        if (i + 1) % 10 == 0:
            print(f"  features: {i + 1}/{len(sample)}")
    feat_shap = np.vstack(feat_shap)
    X_sample = apply_scaler(sample, cfg)

    shap.summary_plot(feat_shap, X_sample, feature_names=features, show=False)
    plt.title(f"{task}: SHAP of the behavioural features on P(positive)")
    plt.tight_layout(); FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / f"shap_features_{task}.png", dpi=150); plt.close()

    imp = pd.Series(np.abs(feat_shap).mean(0), index=features).sort_values(ascending=False)
    signed = pd.Series(feat_shap.mean(0), index=features)
    corr = pd.Series([np.corrcoef(X_sample[:, j], feat_shap[:, j])[0, 1] if X_sample[:, j].std() > 0 else np.nan
                      for j in range(len(features))], index=features)
    lines += ["## 1. Behavioural features (text held fixed)", "",
              f"Beeswarm: `figures/shap_features_{task}.png`. Mean |SHAP| is the average absolute change in "
              "P(positive) attributed to the feature. 'Direction' is the correlation between the scaled feature value "
              "and its SHAP value (+ = higher value pushes towards positive).", "",
              "| Feature | Mean \\|SHAP\\| | Mean SHAP | Direction |", "|---|---|---|---|"]
    lines += [f"| {k} | {imp[k]:.4f} | {signed[k]:+.4f} | {corr[k]:+.2f} |" for k in imp.index]
    total_abs = np.abs(feat_shap).sum(1)
    lines += ["", f"Total |SHAP| over all features per tweet: median {np.median(total_abs):.3f}, "
              f"90th percentile {np.percentile(total_abs, 90):.3f} (on a 0-1 probability scale).", ""]

    # --- 2. text, features fixed ---
    text_rows = sample.groupby("outcome", group_keys=False).apply(lambda g: g.head(3)).reset_index(drop=True)
    masker = shap.maskers.Text(tokenizer)
    html = ["<html><head><meta charset='utf-8'><title>SHAP text</title></head><body>",
            f"<h1>SHAP text explanations: {args.run}</h1>",
            "<p>Red pushes towards the positive class, blue towards negative. Features held fixed at the tweet's values.</p>"]
    for _, row in text_rows.iterrows():
        x = apply_scaler(row.to_frame().T, cfg)
        f = lambda texts, x=x: predict_proba(list(texts), np.repeat(x, len(texts), axis=0))
        sv = shap.Explainer(f, masker)([row["text"]], max_evals=300, silent=True)
        html.append(f"<h3>[{row['outcome']}] p = {row['p1']:.2f}, user {row['user_id']}</h3>")
        html.append(shap.plots.text(sv[0], display=False))
    html.append("</body></html>")
    (RESULTS / f"shap_text_{task}.html").write_text("\n".join(html), encoding="utf-8")
    lines += ["## 2. Text (features held fixed)", "",
              f"Word-level explanations for {len(text_rows)} tweets: `shap_text_{task}.html` (open in a browser).", ""]

    # --- 3. gate values: how much of the feature path reaches the classifier ---
    gates = []
    with torch.no_grad():
        for i in range(0, len(sample), 64):
            chunk = sample.iloc[i:i + 64]
            enc = tokenizer(chunk["text"].tolist(), truncation=True, max_length=cfg["max_len"], padding=True,
                            return_tensors="pt").to(device)
            delta = torch.as_tensor(apply_scaler(chunk, cfg), device=device)
            text = model.base(**enc).last_hidden_state[:, 0]
            encoded = model.feature_encoder(delta)
            gates.append(model.fusion_gate(torch.cat([text, encoded], 1)).mean(1).float().cpu().numpy())
    sample["mean_gate"] = np.concatenate(gates)
    g = sample.groupby("outcome")["mean_gate"].describe()[["count", "mean", "std", "min", "50%", "max"]].round(3)
    lines += ["## 3. Gate values (how much of the feature path is let through, 0 = ignored)", "",
              "| Outcome | n | mean | std | min | median | max |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {k} | {int(r['count'])} | {r['mean']:.3f} | {r['std']:.3f} | {r['min']:.3f} | {r['50%']:.3f} | {r['max']:.3f} |"
              for k, r in g.iterrows()]
    lines += ["",
              f"Overall mean gate {sample['mean_gate'].mean():.3f} (std {sample['mean_gate'].std():.3f}).", ""]

    # --- sampled tweets ---
    lines += ["## Sampled tweets", "", "| Outcome | p | Mean gate | Text |", "|---|---|---|---|"]
    for _, row in sample.sort_values(["outcome", "p1"]).iterrows():
        text = str(row["text"]).replace("|", "\\|").replace("\n", " ")[:160]
        lines.append(f"| {row['outcome']} | {row['p1']:.2f} | {row['mean_gate']:.2f} | {text} |")
    (RESULTS / f"shap_{task}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written to {RESULTS / f'shap_{task}.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="mental_health__length+sentiment+language__s42")
    p.add_argument("--n_examples", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
