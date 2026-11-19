"""
Compares the text-only and text + features models from their saved test predictions (no GPU needed).

  python ml/analysis/compare_models.py            # all tasks, seed 42
  python ml/analysis/compare_models.py --tasks mental_health depression

Writes results/model_comparison.md and results/figures/compare_*.png:
  - tweet- and user-level accuracy / precision / recall / F1 / AUC, confusion matrices
  - prediction confidence: confidence distribution, calibration (reliability diagram, ECE, Brier),
    accuracy by confidence bin
  - user-level ROC and precision-recall curves (Twitter, and Reddit where available)
  - where the two models disagree and which one is right
  - when the features help: performance by how unusual the tweet's behaviour is (feature magnitude)
    and by how many tweets a user has
  - the screener's recall per condition, and the 4-class confusion matrices
"""
import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from features import apply_scaler, load_split  # noqa: E402

RUNS = ROOT / "out" / "v2"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FEATURES = "length+sentiment+language"
LABELS = ["control", "depression", "anxiety", "bipolar"]
MODELS = {"text": "Text only", "feat": "Text + features"}
COLORS = {"text": "#2a78d6", "feat": "#eb6834"}        # categorical slots 1 and 2
POS_C, NEG_C, GRID = "#2a78d6", "#9a9892", "#e6e4df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150})
MD = []


def md(*lines):
    MD.extend(lines)


def table(header, rows):
    md("| " + " | ".join(header) + " |", "|" + "---|" * len(header))
    for r in rows:
        md("| " + " | ".join(str(x) for x in r) + " |")
    md("")


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def run_dir(task, feats, seed):
    return RUNS / f"{task}__{feats}__s{seed}"


def load_pair(task, seed, which="pred_test"):
    """One row per tweet with p_text, p_feat and the thresholds of both models."""
    out, thr = None, {}
    for key, feats in (("text", "none"), ("feat", FEATURES)):
        d = run_dir(task, feats, seed)
        p = pd.read_parquet(d / f"{which}.parquet")
        cols = [c for c in p.columns if c.startswith("p")]
        p = p.rename(columns={c: f"{c}_{key}" for c in cols})
        assert p["tweet_id"].is_unique, f"{d}: duplicate tweet_id"
        out = p if out is None else out.merge(p, on=["tweet_id", "user_id", "y"], how="inner", validate="1:1")
        m = json.loads((d / "metrics.json").read_text())
        if "threshold" in m.get("val_tweet", {}):
            thr[key] = {"tweet": m["val_tweet"]["threshold"], "user": m["val_user"]["threshold"]}
    return out, thr


def user_level(df, cols):
    return df.groupby("user_id").agg({**{c: "mean" for c in cols}, "y": "first"}).reset_index()


def binary_metrics(y, p, t):
    pred = (p >= t).astype(int)
    return {"n": len(y), "accuracy": accuracy_score(y, pred), "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y, pred, average="macro", zero_division=0), "macro_f1": f1_score(y, pred, average="macro"),
            "pos_precision": precision_score(y, pred, zero_division=0), "pos_recall": recall_score(y, pred, zero_division=0),
            "pos_f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "brier": brier_score_loss(y, p), "ece": ece(y, p)}


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return out


def paired_boot_auc(df, col_a, col_b, n_boot=1000, seed=0):
    """Difference in tweet AUC (b - a) with a 95% CI and two-sided p, resampling users."""
    rng = np.random.default_rng(seed)
    groups = {u: idx for u, idx in df.groupby("user_id").indices.items()}
    users = np.array(list(groups))
    y, a, b = df["y"].to_numpy(), df[col_a].to_numpy(), df[col_b].to_numpy()
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[u] for u in rng.choice(users, len(users), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return roc_auc_score(y, b) - roc_auc_score(y, a), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), min(1.0, p)


def confusion_panel(ax, y, pred, labels, title):
    cm = confusion_matrix(y, pred, labels=range(len(labels)))
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(labels)), labels, rotation=30 if len(labels) > 2 else 0, ha="right" if len(labels) > 2 else "center")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10); ax.grid(False)
    return cm


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIGURES / name); plt.close(fig)
    return f"figures/{name}"


# ----------------------------------------------------------------------------------------------------
def binary_task(task, seed, n_boot):
    df, thr = load_pair(task, seed)
    users = user_level(df, ["p1_text", "p1_feat"])
    pretty = "Mental health vs control" if task == "mental_health" else f"{task.capitalize()} vs rest"
    md(f"## {pretty} (seed {seed})", "",
       f"{len(df):,} test tweets from {len(users):,} users; positive rate {df['y'].mean():.2f} per tweet, "
       f"{users['y'].mean():.2f} per user. Thresholds tuned on validation: "
       f"tweet {thr['text']['tweet']:.2f} / {thr['feat']['tweet']:.2f}, user {thr['text']['user']:.2f} / {thr['feat']['user']:.2f} "
       "(text only / + features).", "")

    # --- 1. metrics ---
    mt = {k: binary_metrics(df["y"].to_numpy(), df[f"p1_{k}"].to_numpy(), thr[k]["tweet"]) for k in MODELS}
    mu = {k: binary_metrics(users["y"].to_numpy(), users[f"p1_{k}"].to_numpy(), thr[k]["user"]) for k in MODELS}
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "pos_precision", "pos_recall", "pos_f1", "roc_auc", "brier", "ece"]
    md("### Metrics", "", "Brier and ECE (expected calibration error) are lower-is-better; everything else higher.", "")
    table(["Level", "Model"] + keys,
          [[lvl, MODELS[k]] + [f3(m[k][x]) for x in keys] for lvl, m in (("tweet", mt), ("user", mu)) for k in MODELS])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    show = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc"]
    for ax, (lvl, m) in zip(axes, (("Per tweet", mt), ("Per user", mu))):
        x = np.arange(len(show))
        for i, k in enumerate(MODELS):
            ax.bar(x + (i - 0.5) * 0.36, [m[k][s] for s in show], 0.34, color=COLORS[k], label=MODELS[k], linewidth=0)
        ax.set_xticks(x, [s.replace("_", " ") for s in show], rotation=20); ax.set_ylim(0, 1); ax.set_title(lvl, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower right"); axes[0].set_ylabel("Score")
    fig.suptitle(pretty, fontsize=11)
    md(f"![metrics]({save(fig, f'compare_metrics_{task}.png')})", "")

    # --- 2. confusion matrices ---
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
    for i, (lvl, d, t) in enumerate((("tweet", df, "tweet"), ("user", users, "user"))):
        for j, k in enumerate(MODELS):
            confusion_panel(axes[i, j], d["y"].to_numpy(), (d[f"p1_{k}"] >= thr[k][t]).astype(int).to_numpy(),
                            ["negative", "positive"], f"{MODELS[k]}, per {lvl}")
    fig.suptitle(f"{pretty}: confusion matrices (row-normalised, counts in brackets)", fontsize=11)
    md("### Confusion matrices", "", f"![confusion]({save(fig, f'compare_confusion_{task}.png')})", "")

    # --- 3. confidence and calibration ---
    md("### Prediction confidence and calibration", "",
       "Confidence = 2·|P(positive) − 0.5| (0 = coin flip, 1 = certain). 'Correct' uses the tuned tweet threshold.", "")
    rows, fig = [], plt.figure(figsize=(11, 3.4))
    ax1, ax2, ax3 = fig.subplots(1, 3)
    bins = np.linspace(0, 1, 11)
    for k in MODELS:
        p = df[f"p1_{k}"].to_numpy(); y = df["y"].to_numpy(); conf = 2 * np.abs(p - 0.5)
        correct = ((p >= thr[k]["tweet"]).astype(int) == y)
        rows.append([MODELS[k], f3(conf.mean()), f3(conf[correct].mean()), f3(conf[~correct].mean()),
                     f"{(conf >= 0.5).mean():.1%}", f3(correct[conf >= 0.5].mean()) if (conf >= 0.5).any() else "n/a",
                     f"{(conf < 0.2).mean():.1%}", f3(correct[conf < 0.2].mean())])
        ax1.hist(conf, bins=20, range=(0, 1), histtype="step", lw=2, color=COLORS[k], label=MODELS[k], density=True)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ax2.plot(xs, ys, "o-", color=COLORS[k], lw=2, ms=5, label=f"{MODELS[k]} (ECE {ece(y, p):.3f})")
        cidx = np.clip(np.digitize(conf, np.linspace(0, 1, 6)) - 1, 0, 4)
        ax3.plot([0.1, 0.3, 0.5, 0.7, 0.9][:5], [correct[cidx == b].mean() if (cidx == b).any() else np.nan for b in range(5)],
                 "o-", color=COLORS[k], lw=2, ms=5, label=MODELS[k])
    ax1.set_xlabel("Confidence"); ax1.set_ylabel("Density"); ax1.set_title("Confidence distribution", fontsize=10); ax1.legend()
    ax2.plot([0, 1], [0, 1], "--", color=NEG_C, lw=1); ax2.set_xlabel("Mean P(positive) in bin"); ax2.set_ylabel("Observed positive rate")
    ax2.set_title("Reliability (per tweet)", fontsize=10); ax2.legend(fontsize=8)
    ax3.set_xlabel("Confidence bin"); ax3.set_ylabel("Accuracy"); ax3.set_ylim(0, 1); ax3.set_title("Accuracy by confidence", fontsize=10)
    fig.suptitle(pretty, fontsize=11)
    table(["Model", "Mean conf.", "Conf. when correct", "Conf. when wrong", "Tweets with conf. ≥ 0.5", "Accuracy there",
           "Tweets with conf. < 0.2", "Accuracy there"], rows)
    md(f"![confidence]({save(fig, f'compare_confidence_{task}.png')})", "")

    # --- 4. ROC / PR curves (user level), Twitter + Reddit ---
    erisk = run_dir(task, "none", seed) / "pred_erisk.parquet"
    panels = [("Twitter test, per user", users)]
    if erisk.exists():
        e, _ = load_pair(task, seed, "pred_erisk")
        panels.append(("eRisk Reddit, per user", user_level(e, ["p1_text", "p1_feat"])))
    fig, axes = plt.subplots(1, 2 * len(panels), figsize=(6 * len(panels), 3.6))
    for j, (name, d) in enumerate(panels):
        for k in MODELS:
            fpr, tpr, _ = roc_curve(d["y"], d[f"p1_{k}"]); pr, rc, _ = precision_recall_curve(d["y"], d[f"p1_{k}"])
            axes[2 * j].plot(fpr, tpr, color=COLORS[k], lw=2, label=f"{MODELS[k]} (AUC {roc_auc_score(d['y'], d[f'p1_{k}']):.3f})")
            axes[2 * j + 1].plot(rc, pr, color=COLORS[k], lw=2, label=MODELS[k])
        axes[2 * j].plot([0, 1], [0, 1], "--", color=NEG_C, lw=1); axes[2 * j].set_title(f"ROC: {name}", fontsize=10)
        axes[2 * j].set_xlabel("False positive rate"); axes[2 * j].set_ylabel("True positive rate"); axes[2 * j].legend(fontsize=8, loc="lower right")
        axes[2 * j + 1].axhline(d["y"].mean(), ls="--", color=NEG_C, lw=1); axes[2 * j + 1].set_title(f"PR: {name}", fontsize=10)
        axes[2 * j + 1].set_xlabel("Recall"); axes[2 * j + 1].set_ylabel("Precision"); axes[2 * j + 1].legend(fontsize=8)
    md("### ROC and precision-recall curves (user level)", "", f"![roc]({save(fig, f'compare_roc_{task}.png')})", "")

    # --- 5. disagreement ---
    md("### Where the models disagree", "")
    rows = []
    for lvl, d, t in (("tweet", df, "tweet"), ("user", users, "user")):
        pt = (d["p1_text"] >= thr["text"][t]).astype(int); pf = (d["p1_feat"] >= thr["feat"][t]).astype(int)
        dis = pt != pf; y = d["y"]
        rows.append([lvl, f"{dis.sum():,} ({dis.mean():.1%})", f3((pt[dis] == y[dis]).mean()) if dis.any() else "n/a",
                     f3((pf[dis] == y[dis]).mean()) if dis.any() else "n/a",
                     f3(np.corrcoef(d["p1_text"], d["p1_feat"])[0, 1])])
    table(["Level", "Disagreements", "Text only right", "+ features right", "Correlation of scores"], rows)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for lab, c, name in ((0, NEG_C, "negative"), (1, POS_C, "positive")):
        m = users["y"] == lab
        ax.scatter(users.loc[m, "p1_text"], users.loc[m, "p1_feat"], s=14, color=c, alpha=0.6, label=name, edgecolor="white", linewidth=0.4)
    ax.plot([0, 1], [0, 1], "--", color=NEG_C, lw=1); ax.set_xlabel("Text only: mean P(positive)"); ax.set_ylabel("+ features: mean P(positive)")
    ax.set_title(f"{pretty}: user scores", fontsize=10); ax.legend()
    md(f"![scores]({save(fig, f'compare_user_scores_{task}.png')})", "")

    # --- 6. when do the features help: feature magnitude and user tweet count ---
    cfg = json.loads((run_dir(task, FEATURES, seed) / "config.json").read_text())
    feats = load_split("test", cfg["features"], with_text=False)
    x = apply_scaler(feats, cfg)
    feats["feat_mag"] = np.abs(x).mean(1)
    feats["max_abs_z"] = np.abs(x).max(1)
    df = df.merge(feats[["tweet_id", "feat_mag", "max_abs_z"]], on="tweet_id", how="left", validate="1:1")
    md("### When do the features help?", "",
       "Tweets are binned by how unusual their behavioural features are: the mean |scaled feature| over the "
       "model's 10 features (each scaled with the train mean/std, so 1 ≈ one standard deviation from typical). "
       "ΔAUC = features minus text only, paired bootstrap over users within the bin.", "")
    df["mag_bin"] = pd.qcut(df["feat_mag"], 4, labels=["Q1 (typical)", "Q2", "Q3", "Q4 (unusual)"])
    rows, pts = [], []
    for b, g in df.groupby("mag_bin", observed=True):
        d, lo, hi, p = paired_boot_auc(g, "p1_text", "p1_feat", n_boot=n_boot, seed=seed)
        auc_t, auc_f = roc_auc_score(g["y"], g["p1_text"]), roc_auc_score(g["y"], g["p1_feat"])
        rows.append([b, f"{len(g):,}", f"{g['feat_mag'].min():.2f}–{g['feat_mag'].max():.2f}", f3(g["y"].mean()), f3(auc_t), f3(auc_f),
                     f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", f"{p:.3f}"])
        pts.append((b, auc_t, auc_f, lo, hi, d))
    table(["Feature magnitude", "Tweets", "Range", "Positive rate", "Text AUC", "+ features AUC", "ΔAUC [95% CI]", "p"], rows)

    df["n_tweets"] = df.groupby("user_id")["tweet_id"].transform("size")
    users = users.merge(df.groupby("user_id")["n_tweets"].first(), on="user_id")
    users["n_bin"] = pd.qcut(users["n_tweets"], 3, labels=["few", "medium", "many"])
    rows2 = []
    for b, g in users.groupby("n_bin", observed=True):
        rows2.append([b, len(g), f"{g['n_tweets'].min()}–{g['n_tweets'].max()}", f3(roc_auc_score(g["y"], g["p1_text"])),
                      f3(roc_auc_score(g["y"], g["p1_feat"]))])
    md("User-level AUC by how many test tweets the user has (terciles):", "")
    table(["Tweets per user", "Users", "Range", "Text only user AUC", "+ features user AUC"], rows2)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    xb = np.arange(4)
    for i, k in enumerate(MODELS):
        axes[0].bar(xb + (i - 0.5) * 0.36, [p[1 + i] for p in pts], 0.34, color=COLORS[k], label=MODELS[k], linewidth=0)
    axes[0].set_xticks(xb, [p[0] for p in pts], rotation=15); axes[0].set_ylim(0.5, max(0.8, max(max(p[1], p[2]) for p in pts) + 0.03))
    axes[0].set_ylabel("Tweet AUC"); axes[0].set_title("AUC by feature magnitude", fontsize=10); axes[0].legend(fontsize=8, loc="lower left"); axes[0].grid(axis="x", visible=False)
    axes[1].errorbar(xb, [p[5] for p in pts], yerr=[[p[5] - p[3] for p in pts], [p[4] - p[5] for p in pts]], fmt="o", color=COLORS["feat"],
                     capsize=4, lw=2, ms=6)
    axes[1].axhline(0, color=NEG_C, lw=1, ls="--"); axes[1].set_xticks(xb, [p[0] for p in pts], rotation=15)
    axes[1].set_ylabel("ΔAUC (features − text) [95% CI]"); axes[1].set_title("Feature gain by feature magnitude", fontsize=10)
    fig.suptitle(pretty, fontsize=11)
    md(f"![feature bins]({save(fig, f'compare_feature_bins_{task}.png')})", "")

    # --- 7. per-condition recall of the screener ---
    if task == "mental_health":
        lab = load_split("test", [], with_text=False)[["user_id", "label"]].drop_duplicates("user_id")
        u = users.merge(lab, on="user_id")
        md("### Screener recall per condition", "", "Share of each group's users flagged as 'mental health' at the tuned user threshold, "
           "and the mean user score. The control row is the false-positive rate.", "")
        rows = []
        for c, name in enumerate(LABELS):
            g = u[u["label"] == c]
            rows.append([name, len(g)] + [f"{(g[f'p1_{k}'] >= thr[k]['user']).mean():.2f} (score {g[f'p1_{k}'].mean():.2f})" for k in MODELS])
        table(["Group", "Users", "Text only: flagged (mean score)", "+ features: flagged (mean score)"], rows)
    return {"task": pretty, "tweet": mt, "user": mu}


def fourclass(seed):
    df, _ = load_pair("4class", seed)
    cols = [f"p{i}" for i in range(4)]
    users = user_level(df, [f"{c}_{k}" for c in cols for k in MODELS])
    md(f"## 4-class model (seed {seed})", "", f"{len(df):,} test tweets from {len(users):,} users. Prediction = argmax; "
       "the user prediction is the argmax of the mean probabilities.", "")
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    for i, (lvl, d) in enumerate((("tweet", df), ("user", users))):
        for j, k in enumerate(MODELS):
            pred = d[[f"{c}_{k}" for c in cols]].to_numpy().argmax(1); y = d["y"].to_numpy()
            f1s = f1_score(y, pred, average=None, labels=range(4), zero_division=0)
            rows.append([lvl, MODELS[k], f3(accuracy_score(y, pred)), f3(f1s.mean())] + [f3(v) for v in f1s])
            confusion_panel(axes[i, j], y, pred, LABELS, f"{MODELS[k]}, per {lvl}")
    table(["Level", "Model", "Accuracy", "Macro F1"] + [f"F1 {l}" for l in LABELS], rows)
    fig.suptitle("4-class: confusion matrices (row-normalised, counts in brackets)", fontsize=11)
    md(f"![4class]({save(fig, 'compare_4class_confusion.png')})", "")


def main(args):
    md("# Text-only vs text + features: detailed comparison", "",
       f"Generated by `ml/analysis/compare_models.py` from the saved test predictions (`out/v2/*/pred_test.parquet`, seed {args.seed}). "
       "Thresholds are the ones tuned on validation. Figures are in `results/figures/compare_*.png`.", "")
    summary = []
    for task in args.tasks:
        if task == "4class":
            fourclass(args.seed)
        else:
            summary.append(binary_task(task, args.seed, args.n_boot))
    if summary:
        md("## Summary", "")
        table(["Task", "Level", "Accuracy text / feat", "Macro F1 text / feat", "AUC text / feat", "ECE text / feat"],
              [[s["task"], lvl] + [f"{s[lvl]['text'][m]:.3f} / {s[lvl]['feat'][m]:.3f}" for m in ("accuracy", "macro_f1", "roc_auc", "ece")]
               for s in summary for lvl in ("tweet", "user")])
    (RESULTS / "model_comparison.md").write_text("\n".join(MD) + "\n", encoding="utf-8")
    print(f"Written to {RESULTS / 'model_comparison.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["mental_health", "depression", "anxiety", "bipolar", "4class"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_boot", type=int, default=500)
    main(p.parse_args())
