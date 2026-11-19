"""
Collects out/v2/*/metrics.json into one table (results/summary.md and results/summary.csv).

  python ml/evaluation/summarize.py

Rows are runs; a second table averages over seeds for each (task, features) pair.
Binary tasks: tweet- and user-level ROC-AUC and macro F1 on test (thresholds tuned on val).
4-class: macro F1 (argmax).
"""
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from features import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "out" / "v2"
RESULTS_DIR = PROJECT_ROOT / "results"


def fmt_ci(value, ci):
    if ci is None or any(pd.isna(ci)):
        return f"{value:.3f}"
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def load_runs():
    rows = []
    for metrics_path in sorted(RUNS_DIR.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        m = json.loads(metrics_path.read_text())
        c = json.loads((run_dir / "config.json").read_text())
        t, u = m["test_tweet"], m["test_user"]
        row = {
            "run": run_dir.name, "task": c["task"], "features": c["feature_spec"], "n_features": len(c["features"]),
            "seed": c["args"]["seed"], "best_epoch": m["best_epoch"],
            "tweet_macro_f1": t["macro_f1"], "user_macro_f1": u["macro_f1"],
            "tweet_macro_f1_ci": t.get("macro_f1_ci"), "user_macro_f1_ci": u.get("macro_f1_ci"),
        }
        if c["num_labels"] == 2:
            row.update({
                "tweet_roc_auc": t["roc_auc"], "user_roc_auc": u["roc_auc"],
                "tweet_roc_auc_ci": t.get("roc_auc_ci"), "user_roc_auc_ci": u.get("roc_auc_ci"),
                "tweet_pr_auc": t["pr_auc"], "tweet_threshold": t["threshold"],
                "tweet_macro_f1_at_0.5": m["test_tweet_default"]["macro_f1"],
                "test_users": u["n"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = load_runs()
    if df.empty:
        print(f"No runs found in {RUNS_DIR}")
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    df.drop(columns=[c for c in df.columns if c.endswith("_ci")]).to_csv(RESULTS_DIR / "summary.csv", index=False)

    lines = ["# Results (v2 pipeline)", "",
             "Test set. Thresholds tuned for macro F1 on validation. 95% intervals resample users.", ""]
    for task, group in df.groupby("task", sort=False):
        binary = "tweet_roc_auc" in group and group["tweet_roc_auc"].notna().any()
        lines += [f"## {task}", ""]
        if binary:
            lines += ["| Features | Seed | Tweet ROC-AUC | Tweet macro F1 | User ROC-AUC | User macro F1 |",
                      "|---|---|---|---|---|---|"]
            for _, r in group.sort_values(["features", "seed"]).iterrows():
                lines.append(f"| {r['features']} | {r['seed']} | {fmt_ci(r['tweet_roc_auc'], r['tweet_roc_auc_ci'])} | "
                             f"{fmt_ci(r['tweet_macro_f1'], r['tweet_macro_f1_ci'])} | "
                             f"{fmt_ci(r['user_roc_auc'], r['user_roc_auc_ci'])} | "
                             f"{fmt_ci(r['user_macro_f1'], r['user_macro_f1_ci'])} |")
            metric_cols = ["tweet_roc_auc", "tweet_macro_f1", "user_roc_auc", "user_macro_f1"]
        else:
            lines += ["| Features | Seed | Tweet macro F1 | User macro F1 |", "|---|---|---|---|"]
            for _, r in group.sort_values(["features", "seed"]).iterrows():
                lines.append(f"| {r['features']} | {r['seed']} | {fmt_ci(r['tweet_macro_f1'], r['tweet_macro_f1_ci'])} | "
                             f"{fmt_ci(r['user_macro_f1'], r['user_macro_f1_ci'])} |")
            metric_cols = ["tweet_macro_f1", "user_macro_f1"]

        seeds = group.groupby("features")[metric_cols].agg(["mean", "std", "count"])
        if (seeds.xs("count", axis=1, level=1) > 1).any().any():
            lines += ["", "Mean ± std over seeds:", "", "| Features | n | " + " | ".join(metric_cols) + " |",
                      "|---|---|" + "---|" * len(metric_cols)]
            for feats, r in seeds.iterrows():
                cells = [f"{r[(c, 'mean')]:.3f} ± {0 if pd.isna(r[(c, 'std')]) else r[(c, 'std')]:.3f}" for c in metric_cols]
                lines.append(f"| {feats} | {int(r[(metric_cols[0], 'count')])} | " + " | ".join(cells) + " |")
        lines.append("")

    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
