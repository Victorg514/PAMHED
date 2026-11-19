"""
Metrics shared by the training script and the summary script.

- Positive-class F1 is reported but never used to choose anything: always predicting the positive
  class scores well on it. Model selection uses ROC-AUC; thresholds are tuned
  for macro F1 on the validation split and then applied unchanged to test.
- User-level scores average each user's tweet probabilities, as in eRisk-style evaluation.
- Confidence intervals resample USERS (not tweets), because tweets from one user are not independent.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def tune_threshold(y, p) -> float:
    """Threshold on P(positive) that maximizes macro F1."""
    scores = [f1_score(y, (p >= t).astype(int), average="macro", zero_division=0) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def binary_scores(y, p, threshold) -> dict:
    pred = (p >= threshold).astype(int)
    both_classes = len(np.unique(y)) == 2
    return {
        "roc_auc": roc_auc_score(y, p) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y, p) if both_classes else float("nan"),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "pos_f1": f1_score(y, pred, average="binary", zero_division=0),
        "pos_precision": precision_score(y, pred, zero_division=0),
        "pos_recall": recall_score(y, pred, zero_division=0),
        "pred_pos_rate": float(pred.mean()),
        "true_pos_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "n": int(len(y)),
    }


def multiclass_scores(y, proba) -> dict:
    pred = proba.argmax(axis=1)
    out = {"macro_f1": f1_score(y, pred, average="macro", zero_division=0), "n": int(len(y))}
    try:
        out["roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", average="macro")
    except ValueError:
        out["roc_auc_ovr"] = float("nan")
    for c, f in enumerate(f1_score(y, pred, average=None, labels=range(proba.shape[1]), zero_division=0)):
        out[f"f1_class{c}"] = f
    return out


def to_user_level(df: pd.DataFrame, prob_cols) -> pd.DataFrame:
    """Mean probability per user. df has user_id, y and the probability columns."""
    return df.groupby("user_id").agg({**{c: "mean" for c in prob_cols}, "y": "first"}).reset_index()


def bootstrap_ci(df: pd.DataFrame, score_fn, n_boot=1000, seed=0) -> list:
    """95% interval of score_fn(df_resampled), resampling users with replacement."""
    rng = np.random.default_rng(seed)
    groups = {u: idx for u, idx in df.groupby("user_id").indices.items()}
    users = np.array(list(groups))
    scores = []
    for _ in range(n_boot):
        sample = rng.choice(users, size=len(users), replace=True)
        idx = np.concatenate([groups[u] for u in sample])
        try:
            s = score_fn(df.iloc[idx])
        except ValueError:
            continue
        if not np.isnan(s):
            scores.append(s)
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))] if scores else [float("nan")] * 2


def evaluate_predictions(val: pd.DataFrame, test: pd.DataFrame, num_labels: int, n_boot=1000) -> dict:
    """
    val/test: one row per tweet with user_id, y and p0..p{k-1}.
    Returns tweet-level and user-level metrics on test (thresholds tuned on val), plus val metrics.
    """
    prob_cols = [f"p{i}" for i in range(num_labels)]
    out = {}
    if num_labels == 2:
        t_tweet = tune_threshold(val["y"].to_numpy(), val["p1"].to_numpy())
        out["val_tweet"] = binary_scores(val["y"].to_numpy(), val["p1"].to_numpy(), t_tweet)
        out["test_tweet_default"] = binary_scores(test["y"].to_numpy(), test["p1"].to_numpy(), 0.5)
        out["test_tweet"] = binary_scores(test["y"].to_numpy(), test["p1"].to_numpy(), t_tweet)

        uval, utest = to_user_level(val, prob_cols), to_user_level(test, prob_cols)
        t_user = tune_threshold(uval["y"].to_numpy(), uval["p1"].to_numpy())
        out["val_user"] = binary_scores(uval["y"].to_numpy(), uval["p1"].to_numpy(), t_user)
        out["test_user"] = binary_scores(utest["y"].to_numpy(), utest["p1"].to_numpy(), t_user)

        if n_boot:
            out["test_tweet"]["roc_auc_ci"] = bootstrap_ci(test, lambda d: roc_auc_score(d["y"], d["p1"]), n_boot)
            out["test_tweet"]["macro_f1_ci"] = bootstrap_ci(
                test, lambda d: f1_score(d["y"], (d["p1"] >= t_tweet).astype(int), average="macro", zero_division=0), n_boot)
            out["test_user"]["roc_auc_ci"] = bootstrap_ci(utest, lambda d: roc_auc_score(d["y"], d["p1"]), n_boot)
            out["test_user"]["macro_f1_ci"] = bootstrap_ci(
                utest, lambda d: f1_score(d["y"], (d["p1"] >= t_user).astype(int), average="macro", zero_division=0), n_boot)
    else:
        out["val_tweet"] = multiclass_scores(val["y"].to_numpy(), val[prob_cols].to_numpy())
        out["test_tweet"] = multiclass_scores(test["y"].to_numpy(), test[prob_cols].to_numpy())
        utest = to_user_level(test, prob_cols)
        out["test_user"] = multiclass_scores(utest["y"].to_numpy(), utest[prob_cols].to_numpy())
        if n_boot:
            out["test_tweet"]["macro_f1_ci"] = bootstrap_ci(
                test, lambda d: f1_score(d["y"], d[prob_cols].to_numpy().argmax(1), average="macro", zero_division=0), n_boot)
            out["test_user"]["macro_f1_ci"] = bootstrap_ci(
                utest, lambda d: f1_score(d["y"], d[prob_cols].to_numpy().argmax(1), average="macro", zero_division=0), n_boot)
    return out
