"""
Shared feature definitions and data loading for training, evaluation, ablation and SHAP.

Everything that turns data/final.parquet into model inputs lives here, so the training script,
the evaluation scripts and the SHAP notebook cannot drift apart (see FEATURE_FIXES.md).
"""
import pathlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "final.parquet"

LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
TASKS = ["mental_health", "depression", "anxiety", "bipolar", "4class"]
CLIP = 5.0         # standardized features are clipped to [-CLIP, CLIP]
WINSOR = (0.5, 99.5)  # raw features are first clipped to these train percentiles (heavy tails)

# One representative per feature family: no shifted/rescaled copies of the same signal (fix 3)
FEATURE_GROUPS = {
    "length":     ["len_z"],
    "sentiment":  ["sent_compound_z", "neg", "pos", "emotion_volatility_10tw", "polarity_shift"],
    "timing":     ["circadian_deviation", "hour_sin", "hour_cos", "late_night"],
    "activity":   ["tpd_7d_dev", "tpd_30d_dev", "burst_posting", "silence_period"],
    "language":   ["self_focus", "exclamation_rate", "isolation_rate", "mention_count"],
    "indicators": ["manic_indicator", "depressive_indicator"],
}
DEFAULT_GROUPS = list(FEATURE_GROUPS)

# Every numeric feature in the v2 parquet, including the redundant ones (only used by the ablation)
ALL_FEATURES = [
    "len_val", "len_dev", "len_z", "neg", "neu", "pos", "compound", "sent_compound_dev", "sent_compound_z",
    "circadian_deviation", "hour_sin", "hour_cos", "late_night", "burst_posting", "silence_period",
    "tpd_3d", "tpd_3d_dev", "tpd_7d", "tpd_7d_dev", "tpd_14d", "tpd_14d_dev", "tpd_30d", "tpd_30d_dev",
    "emotion_volatility_10tw", "polarity_shift", "self_focus", "exclamation_rate", "isolation_rate",
    "mention_count", "anomaly_score", "manic_indicator", "depressive_indicator",
]


def resolve_features(spec: str) -> list:
    """
    Turns a feature spec into an ordered feature list.
      "none"                 -> [] (text-only model)
      "default"              -> all FEATURE_GROUPS
      "all"                  -> ALL_FEATURES
      "timing,activity"      -> those groups
      "default,-length"      -> default groups minus 'length' (leave-one-group-out)
    """
    spec = spec.strip().lower()
    if spec == "none":
        return []
    if spec == "all":
        return list(ALL_FEATURES)
    groups = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        if token == "default":
            groups += [g for g in DEFAULT_GROUPS if g not in groups]
        elif token.startswith("-"):
            groups = [g for g in groups if g != token[1:]]
        elif token in FEATURE_GROUPS:
            if token not in groups:
                groups.append(token)
        else:
            raise ValueError(f"Unknown feature group '{token}'. Known: {list(FEATURE_GROUPS)}")
    return [f for g in groups for f in FEATURE_GROUPS[g]]


def num_labels_for(task: str) -> int:
    return 4 if task == "4class" else 2


def task_labels(task: str, label: np.ndarray) -> np.ndarray:
    """Maps the 4-way user label to the task's label: binary tasks use 1 = positive class."""
    label = np.asarray(label)
    if task == "4class":
        return label.astype(int)
    if task == "mental_health":
        return (label != LABEL_MAP["control"]).astype(int)
    return (label == LABEL_MAP[task]).astype(int)


def load_split(split: str, features: list, with_text: bool = True, data_file=DATA_FILE) -> pd.DataFrame:
    """Reads one split with only the needed columns (the full parquet does not fit in RAM comfortably)."""
    columns = ["tweet_id", "user_id", "label"] + (["text"] if with_text else []) + list(features)
    return pq.read_table(data_file, columns=columns, filters=[("split", "==", split)]).to_pandas()


def fit_scaler(train_df: pd.DataFrame, features: list) -> dict:
    """
    Statistics from the TRAIN split only (fix 6). Raw values are clipped to the train percentiles in
    WINSOR before mean/std are computed, so a few extreme rows cannot squash a whole feature.
    """
    x = train_df[features].to_numpy(dtype=np.float64)
    if x.shape[1] == 0:
        return {"features": [], "feature_lo": [], "feature_hi": [], "feature_mean": [], "feature_std": []}
    lo, hi = np.percentile(x, WINSOR, axis=0)
    x = np.clip(x, lo, hi)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"features": list(features), "feature_lo": lo.tolist(), "feature_hi": hi.tolist(),
            "feature_mean": x.mean(axis=0).tolist(), "feature_std": std.tolist()}


def apply_scaler(df: pd.DataFrame, scaler: dict) -> np.ndarray:
    feats = scaler["features"]
    if not feats:
        return np.zeros((len(df), 0), dtype=np.float32)
    x = np.clip(df[feats].to_numpy(dtype=np.float64), scaler["feature_lo"], scaler["feature_hi"])
    x = (x - np.asarray(scaler["feature_mean"])) / np.asarray(scaler["feature_std"])
    return np.clip(np.nan_to_num(x), -CLIP, CLIP).astype(np.float32)
