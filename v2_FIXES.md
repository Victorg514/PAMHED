# Feature fixes

Known problems with the behavioural features and how they are selected for the delta models,
in rough order of importance. Features are built in `ml/preprocessing/merge_blip.py`.
In v1, selection happened in `select_best_features()` in `ml/legacy/train_binary_delta_manual.py`; in v2 the
feature set is fixed by `ml/features.py` and the ablation.

## v2 status (2026-11-13)

All fixes below are implemented in the v2 pipeline. The v1 data and cached predictions are kept in
`data/backup_v1/`.

| Fix | Where |
|---|---|
| 1, 2, 3, 4, 8: explicit feature groups, no MI, no duplicates, hour sin/cos and indicators | `ml/features.py` |
| 5: circular `circadian_deviation` | `merge_blip.py` |
| 6: scaling fit on train only (plus clipping to train percentiles) | `ml/features.py` |
| 7: counts divided by word count (`exclamation_rate`, `isolation_rate`) | `merge_blip.py` |
| 9: feature-group ablation | `ml/analysis/feature_ablation.py`, `results/feature_ablation_*.md` |
| 10: baseline and later tweets use the same cleaned text, without the caption | `merge_blip.py` |
| 11: `mention_count` counted on the raw text | `merge_blip.py` |
| Single class balancing, val-tuned thresholds, macro F1 / AUC, per-user metrics | `ml/training/train.py`, `ml/evaluation/metrics.py` |

Also fixed in v2:
- **Retweets were assigned to the wrong user (v1 and the first v2 build).** `merge_blip.py` keyed
  tweets on the author (`tw["user"]["profile_id"]`), not on the scraped timeline owner. Accounts
  retweeted by many scraped users became fake "users" (662 of 3,153, 16% of rows) with a near-random
  label and a baseline mixing many people; the 5% duplicate rows were a symptom. Fixed on 2026-11-14:
  tweets go to the timeline owner, retweets are dropped, users in more than one condition file are
  dropped. All runs before this fix are in `out/v2_retweet_bug/`.
- **`len_z` was broken in v1.** Users whose baseline tweets all had the same length got z-scores
  in the billions, giving the feature a standard deviation of 250 million, so every normal value
  became about 0. v2 sets a minimum for the baseline standard deviation (10 characters for length,
  0.05 for sentiment).
- Rolling and gap features include the baseline tweets, so the first later tweets no longer get
  fake "burst" flags or undercounted posting rates.
- `merge_blip.py` crashed on a missing `_try_langdetect` method.
- After the retweet fix the dataset has 2,553 users (512 in test); real v1 users keep their v1 split.
  v1 and v2 results are therefore not directly comparable (see `results/RESULTS_NOTES.md`).

**Ablation result (validation split, features only, clean data, seed 42):** the best set at the user
level is length + sentiment + language (10 features). Per tweet, all 20 features score higher for
mental health (the activity group carries some signal on the clean data), but at the user level the
10-feature set matches or beats them on every task. Language (mostly `self_focus`, the first-person
pronoun rate) is the strongest group: removing it costs the most.

| Task | Feature set | Val tweet AUC | Val user AUC |
|---|---|---|---|
| Mental health | all 20 (default) | **0.715** | 0.783 |
| Mental health | length + sentiment + language | 0.687 | **0.799** |
| Depression | all 20 (default) | 0.565 | 0.587 |
| Depression | length + sentiment + language | **0.582** | **0.622** |

Raw `len_val`/`tpd_*` values describe the account type, not a change from the user's baseline,
so they stay out (fix 9).

**Final result (test set, 2026-11-15):** with the text model, the 10 features give no significant
user-level gain on any Twitter task (paired bootstrap; anxiety checked over 3 seeds). On eRisk Reddit
(64 depressed / 622 control users, Twitter scaling and thresholds, nothing fit on Reddit) they raise
per-post AUC by about +0.025 (p < 0.001) but not user AUC (0.84–0.85 either way). Details in
`results/RESULTS_NOTES.md`.

How to run: `python ml/run_queue.py` (see its docstring for phases), then
`python ml/evaluation/summarize.py`. Explain a run with `notebooks/SHAP_v2.ipynb`.

## v1 state (what the fixes below refer to)

- 28 numeric features reach the model: every `float64`/`int64` column in `data/final.parquet`
  except IDs and labels.
- Each user's baseline is the first 40% of their timeline (minimum 10 posts).
  Features are only computed for posts after the baseline.
- `MAX_FEATURES_TO_SELECT = 50` with only 28 candidates, so mutual information (MI) selects nothing.
  It only decides the column order.

Test F1 from the cached predictions (`cache/`, 107,839 test tweets):

| Task | Text only | Text + features |
|---|---|---|
| Mental health vs. control, positive-class F1 | 0.674 | 0.693 |
| Depression vs. rest, positive-class F1 | 0.314 | 0.324 |
| Depression vs. rest, macro F1 | 0.435 | 0.371 |
| 4-class, macro F1 | 0.304 | 0.305 |

The features add about 0.01–0.02 F1. These are single runs, so that is close to seed noise.

## Fixes

### 1. Save the feature list with the model (no retraining needed)
The SHAP notebook and the evaluation scripts recompute MI to recover the column order.
Any change to the data, the sklearn version or the column order silently swaps features between
columns, and nothing raises an error.
**Fix:** write `selected_features` into `config.json` at training time and read it everywhere else.
Patch the existing `out/*_manual/config.json` files with the current order.

### 2. Replace MI top-50 with a fixed, explicit feature list
MI with a limit of 50 doesn't remove anything. It is extra code whose only effect is the column order.
**Fix:** use a documented feature list. Or, if selection is wanted, use a real limit tuned on
validation F1.

### 3. Remove redundant features
MI scores each feature on its own, so it cannot see duplicates:
- `neg + neu + pos ≈ 1`, so `neu` is redundant.
- `len_val`, `len_dev` and `len_z` are shifted or rescaled copies of each other within a user.
- `tpd_Xd` and `tpd_Xd_dev` differ only by a per-user constant, and the 3/7/14/30-day windows
  are highly correlated.
- `anomaly_score` = |len_z| + |sent_compound_z| + |circadian_deviation|.

**Fix:** drop `neu` and `anomaly_score`, and keep one version per feature family. Prefer the
`_dev`/`_z` versions, since the thesis is about deviation from a personal baseline.
Duplicates also make SHAP split credit between features arbitrarily.

### 4. Stop dropping features by accident
The dtype filter `dtype in ['float64','int64']` silently excludes:
- `hour` (`int32`)
- `manic_indicator` and `depressive_indicator` (`bool`)

**Fix:** use an explicit list. Add the two indicators as 0/1, and add the hour as
`sin(2π·hour/24)` and `cos(2π·hour/24)`.

### 5. Make `circadian_deviation` circular
`(hour − baseline_median) / hour_std` treats the hour as a straight line. For a user whose median
hour is 23:00, a post at 01:00 scores −22/std, even though it is only 2 hours later. This hits
night owls hardest, a key group for depression and bipolar.
**Fix:** wrap the difference with `d = (hour − median + 12) % 24 − 12`, and use a circular
median and standard deviation for the baseline. This needs re-running preprocessing.

### 6. Fit the scaler on train only
`StandardScaler` is fit on all users before the train/val/test split (`merge_blip.py`),
so test statistics leak into training.
**Fix without re-running preprocessing:** global standardisation is only a shift and a rescale,
so fitting a new `StandardScaler` on the train split in the training script and applying it to all
splits gives exactly the same result as fitting on train from the raw data.
Leave `len_z`, `sent_compound_z`, `circadian_deviation` and `anomaly_score` unscaled, as now.

### 7. Normalise count features by length
`exclamation_count`, `mention_count` and `isolation_score` are raw counts, so they partly measure
post length again. `self_focus` already divides by the word count.
**Fix:** divide by the number of words. This can be recomputed from `text` when loading the data.

### 8. Mark binary features as discrete in MI (only if MI is kept)
`late_night`, `burst_posting`, `silence_period` and `polarity_shift` are 0/1 values but are scored
as continuous, which makes their MI noisy.
**Fix:** pass `discrete_features` to `mutual_info_classif`.

### 9. Check for account-type confounding
SHAP shows `len_dev` dominating, at about 4× the next feature. The 20 most confident
"mental_health" predictions include pharma and conference marketing tweets at 99.9%. The features
may be recognising the type of account rather than the user's mental state.
**Fix:** run a feature-group ablation:
1. **On CPU:** logistic regression or LightGBM per group (length, posting rate, sentiment, timing,
   language), evaluated per user.
2. **On GPU:** 2–3 full runs: text only, all features, and the pruned set.

Keep a group only if it adds F1 over text alone.

### 10. Compute the baseline and later posts from the same text
In `process_user_timeline()` in `merge_blip.py`, the two parts of each timeline see different text:
- **Baseline statistics** (`len_mu`, `sent_compound_mu`) use the **raw** tweet text, with URLs and
  @handles and no caption.
- **Later posts** use **cleaned** text: URLs and @handles removed, then `" [IMG_CAP] " + caption`
  appended.

So `len_dev`/`len_z` partly measure "URLs removed, and did this tweet have an image?" rather than a
real change in posting behaviour. The sentiment deviations are affected the same way. `len_dev` is
the top feature in SHAP, so this matters.
**Fix:** compute every feature, baseline included, from the cleaned tweet text *without* the
caption. Use the caption only as model input. Needs re-running preprocessing.

### 11. `mention_count` is almost constant
`_clean_text()` removes `@handles` before `mention_count` counts `@`, so 99.8% of rows have the
same value.
**Fix:** count mentions on the raw text, or drop the feature.

## Related, not about features: decision threshold and class balancing

The bigger F1 problem is how the models turn probabilities into predictions.
- **Depression delta model:** it predicts the positive class for 74% of test tweets, but only 19%
  are positive. It misclassifies 63,622 of 87,867 control tweets, which explains the drop in macro F1.
- **Likely cause:** the manual scripts correct for class imbalance twice, with a
  `WeightedRandomSampler` and a class-weighted loss.

Fixes:
- **Tune the threshold on validation, not test:** `ml/legacy/tune_thresholds.py` (v1); built into `train.py` in v2.
- **Balance only once:** keep either the sampler or the loss weights. Needs retraining.

### Threshold tuning results
Run on 2026-10-14. Thresholds were picked on 10,000 random validation tweets and applied to the
cached test predictions.

| Task | Model | Macro F1 at 0.5 | Macro F1 at tuned threshold | Tuned threshold |
|---|---|---|---|---|
| Mental health | text only | 0.602 | 0.603 | 0.53 |
| Mental health | + features | 0.605 | 0.606 | 0.60 |
| Depression | text only | 0.435 | **0.524** | 0.94 |
| Depression | + features | 0.371 | **0.517** | 0.90 |

- **Mental health:** the threshold barely matters.
- **Depression:** tuning gains about 0.09–0.15 macro F1. At the tuned thresholds the delta model is
  *worse* than text only (0.517 vs 0.524), so the features don't help depression.

### Positive-class F1 can't measure these models
A classifier that always predicts the positive class scores well on positive-class F1:
- **Mental health:** about 59% of test tweets are positive, so always-positive scores
  **0.739**, higher than either model (0.674 / 0.693).
- **Depression:** about 19% are positive, so always-positive scores **0.312**, the same as
  text only (0.314).

Tuning for positive-class F1 on validation picks thresholds like 0.02, which means predicting
nearly everything as positive. **Report macro F1 and ROC/PR-AUC instead.**

Test ROC-AUC, measured per tweet:

| Task | Text only | + features |
|---|---|---|
| Mental health | 0.650 | 0.656 |
| Depression | 0.557 | 0.559 |

Per tweet, depression is close to chance. Most single tweets from depressed users are not about
depression. Grouping predictions per user (for example, the mean probability over a user's
tweets) is the standard eRisk setup and will probably matter more than any feature fix.

## Retraining

Fixes 2–7, 10 and 11 change the model's input, so the delta models must be retrained.
Fixes 5, 10 and 11 also need `merge_blip.py` to be run again. If preprocessing is re-run anyway,
do fix 6 there too. Retraining needs a GPU: the dataset has 361k training rows, and
fine-tuning RoBERTa on CPU would take days.

Order of work:
1. Fix 1, and switch reported metrics to macro F1 and AUC, with val-tuned thresholds
   (no retraining). Threshold tuning is done.
2. Fixes 2–4, 6, 7 in a shared feature function used by the training scripts, the evaluation
   scripts and the SHAP notebook.
3. The CPU part of the ablation (fix 9) to settle the final feature list.
4. Retrain `delta_binary_mental_health_manual` and `delta_binary_depression_manual` once, then
   re-run SHAP.
