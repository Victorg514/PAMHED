# v2 results notes

Notes for writing the paper. Last updated 2026-11-15 (final runs on the clean data). The feature fixes
behind these results are in `FEATURE_FIXES.md`. The raw tables are in `results/summary.md` (Twitter),
`results/erisk_cross_platform.md` (Reddit) and `results/paired_bootstrap_s{42,43,44}.md`.

> **Data history.** A first round of v2 results was computed on data with a retweet bug:
> `merge_blip.py` keyed tweets on the *author* (`tw["user"]["profile_id"]`) instead of the timeline
> owner. Retweets therefore went to the original poster, and popular accounts retweeted by many scraped
> users became fake "users": 662 of 3,153 users were assembled from 10+ different timelines (81,679 rows,
> 16% of the data, 132 of them in the test set), with a near-random label and a baseline that mixes many
> people. The same bug is in v1, so the paper's original Table 1 counts and all v1 results are affected.
>
> Fix: every tweet is assigned to the timeline owner, retweets are dropped (someone else's words), and
> users that appear in more than one condition file are dropped. **All numbers below are from the fixed
> data.** The superseded runs and tables are kept for reference only in `out/v2_retweet_bug/`,
> `results/retweet_bug/` and `data/backup_v2_retweet_bug/`; do not report them.

## Setup

- **Data (after the fix):** 2,553 Twitter users with at least 50 own tweets (retweets excluded) that
  pass the language screen. 449,883 post-baseline tweets: train 314,842 / val 42,303 / test 92,738.

  | Split | Control | Depression | Anxiety | Bipolar | Total users |
  |---|---|---|---|---|---|
  | Train | 911 | 291 | 288 | 291 | 1,781 |
  | Val | 133 | 42 | 40 | 45 | 260 |
  | Test | 254 | 85 | 87 | 86 | 512 |

  Each user's first 40% of tweets (at least 10) is their personal baseline and is not scored.
  Preprocessing: 643,925 retweets dropped; 54 timelines had no own tweets; 80 users were in two condition
  files and were dropped. 431 of the 512 test users were also in the previous test set.
- **Models:** one binary model per task, all trained independently on all users (parallel, not a cascade):
  - mental health vs control (the screener)
  - depression vs rest
  - anxiety vs rest
  - bipolar vs rest

  plus one 4-class model (control / depression / anxiety / bipolar, the paper's Table 2).
  Each model is `mental/mental-roberta-base` on the tweet text plus the image caption, optionally
  combined with 10 behavioural features through a learned gate. The 10 features are length, sentiment
  and language, chosen by a CPU ablation on the validation split (`results/feature_ablation_*.md`).
  The text-only baseline is the same network without features.
- **Training:** 3 epochs maximum. The best epoch is chosen by validation ROC-AUC (macro F1 for 4-class).
  Class imbalance is corrected once, through the loss weights. Seed 42; anxiety also seeds 43 and 44.
  About 88 min per run on the GPU.
- **Evaluation:**
  - The decision threshold is tuned for macro F1 on validation, then applied unchanged to test.
  - A user-level score is the mean probability over the user's tweets.
  - 95% intervals come from resampling users (1000 bootstrap samples).
  - Positive-class F1 is not used: always predicting positive beats the models on it.

## Twitter test set (seed 42)

| Task | Model | Tweet AUC | Tweet macro F1 | User AUC [95% CI] | User macro F1 [95% CI] |
|---|---|---|---|---|---|
| Mental health vs control | text only | 0.749 | 0.677 | 0.841 [0.805, 0.873] | 0.762 [0.722, 0.797] |
| Mental health vs control | + features | 0.757 | 0.683 | 0.837 [0.800, 0.871] | 0.744 [0.705, 0.781] |
| Depression vs rest | text only | 0.618 | 0.543 | 0.666 [0.614, 0.720] | 0.570 [0.521, 0.620] |
| Depression vs rest | + features | 0.620 | 0.542 | 0.665 [0.609, 0.717] | 0.531 [0.487, 0.575] |
| Anxiety vs rest | text only | 0.656 | 0.566 | 0.700 [0.641, 0.754] | 0.612 [0.561, 0.664] |
| Anxiety vs rest | + features | 0.642 | 0.561 | 0.700 [0.639, 0.757] | 0.588 [0.538, 0.639] |
| Bipolar vs rest | text only | 0.654 | 0.568 | 0.706 [0.651, 0.759] | 0.600 [0.552, 0.648] |
| Bipolar vs rest | + features | 0.653 | 0.572 | 0.693 [0.637, 0.745] | 0.596 [0.549, 0.643] |

4-class (control / depression / anxiety / bipolar; chance macro F1 is about 0.25):

| Model | Tweet macro F1 [95% CI] | User macro F1 [95% CI] |
|---|---|---|
| text only | 0.349 [0.330, 0.365] | 0.399 [0.355, 0.439] |
| + features | 0.353 [0.331, 0.372] | 0.372 [0.335, 0.413] |

Anxiety over three seeds (mean ± std):

| Model | Tweet AUC | Tweet macro F1 | User AUC | User macro F1 |
|---|---|---|---|---|
| text only | 0.652 ± 0.004 | 0.564 ± 0.004 | 0.703 ± 0.003 | 0.613 ± 0.014 |
| + features | 0.650 ± 0.011 | 0.567 ± 0.007 | 0.706 ± 0.006 | 0.598 ± 0.008 |

Observations:
- **Fixing the retweet bug is the largest improvement.** Mental health user AUC is 0.84 (0.76 on the
  buggy data); depression, anxiety and bipolar all gain about 0.05–0.07 user AUC. Label noise from the
  fake users was holding every model back.
- **User-level scoring helps every task.** Mental health: 0.75 per tweet → 0.84 per user.
- **The behavioural features do not improve the models on Twitter.** No task shows a user-level gain:
  the differences are within ±0.013 AUC and none is significant (paired bootstrap below). Per tweet,
  the features give mental health a small, borderline gain (+0.008 AUC, p = 0.10).
- **The anxiety gain seen on the buggy data does not replicate.** Over three seeds the user AUC
  difference is +0.000, +0.009 and +0.001.
- **User macro F1 tends to be lower with features** (significant for depression, p = 0.03, and for
  anxiety seed 44). This is the validation-tuned threshold moving, not worse ranking (AUC is equal).
- **Depression remains the hardest condition** (user AUC 0.67). Most tweets from depressed users are
  not about depression.
- **Most models peak early.** The best epoch was epoch 1 for 8 of 14 runs, including every + features
  run for mental health, depression and bipolar; a lower learning rate or fewer epochs may help.
- **Anxiety and bipolar reuse the feature set** chosen on mental health and depression. This keeps all
  four models comparable, but may not be the best set for those two conditions.

### Paired bootstrap

Script: `ml/evaluation/paired_bootstrap.py`, output in `results/paired_bootstrap_s{42,43,44}.md`. Each
bootstrap sample draws test users and scores both models on that same sample. The difference is
features minus text only. This covers test-set sampling noise; the anxiety seeds cover training noise.

| Task | User AUC difference [95% CI] | p | User macro F1 difference [95% CI] | p |
|---|---|---|---|---|
| Mental health | −0.004 [−0.018, +0.010] | 0.63 | −0.018 [−0.039, +0.005] | 0.14 |
| Depression | −0.002 [−0.020, +0.017] | 0.85 | −0.038 [−0.076, −0.003] | 0.03 |
| Anxiety, seed 42 | +0.000 [−0.018, +0.023] | 1.00 | −0.024 [−0.068, +0.016] | 0.29 |
| Anxiety, seed 43 | +0.009 [−0.007, +0.025] | 0.32 | +0.004 [−0.037, +0.043] | 0.83 |
| Anxiety, seed 44 | +0.001 [−0.010, +0.013] | 0.83 | −0.025 [−0.048, −0.002] | 0.03 |
| Bipolar | −0.013 [−0.030, +0.003] | 0.13 | −0.004 [−0.030, +0.021] | 0.78 |

Per tweet (AUC): mental health +0.008 [−0.001, +0.017], p = 0.10; depression +0.002, p = 0.65;
anxiety −0.014 (p = 0.02), +0.011 (p = 0.08), −0.001 (p = 0.76) for seeds 42/43/44; bipolar −0.001, p = 0.91.

### Detailed comparison (`results/model_comparison.md`, `figures/compare_*.png`)

Script: `ml/analysis/compare_models.py`, from the saved predictions (seed 42). Covers accuracy /
precision / recall / F1 per tweet and per user, confusion matrices, confidence and calibration, ROC and
PR curves, disagreements, and when the features help.

- **Calibration is the features' clearest advantage.** The + features model is better calibrated in
  7 of 8 task/level rows (expected calibration error; e.g. mental health per tweet 0.081 vs 0.127,
  Brier 0.208 vs 0.222). It makes fewer very confident predictions, and its confident predictions are
  right more often (mental health: 51% of tweets have confidence >= 0.5 with accuracy 0.79, against 62%
  with accuracy 0.76 for text only).
- **The feature gain grows with how unusual the behaviour is (mental health).** Binning test tweets by
  the mean |scaled feature|, ΔAUC is +0.004, +0.007, +0.010, +0.015 from the most typical to the most
  unusual quartile (p = 0.04 in the top quartile). The text-only model is weakest exactly there
  (0.735 vs 0.75 elsewhere). Depression shows the same direction but is not significant; anxiety is the
  reverse (features hurt per tweet); bipolar is flat.
- **Disagreements are rare at the user level** (5–9% of users; score correlation 0.96–0.97). Per tweet
  the two models disagree on 10–15% of tweets; the + features model wins those for mental health (52%),
  depression (65%) and bipolar (53%), the text-only model for anxiety (62%).
- **The screener catches the three conditions about equally** (user recall 0.71–0.78; anxiety easiest),
  at a 22–26% false-positive rate on controls.
- **More tweets per user = better user AUC** (mental health 0.77 / 0.86 / 0.89 for the tercile with
  the fewest / middle / most test tweets). Supports pooling and the eRisk result.
- **4-class:** both models are weak (user macro F1 0.40 text only, 0.37 + features). Control is
  recognised (F1 0.73); the three conditions are confused with each other. The + features model
  collapses depression into bipolar (user F1 0.11 for depression, 0.41 for bipolar).

### SHAP (`results/shap_mental_health.md`, `figures/shap_features_mental_health.png`, `shap_text_mental_health.html`)

Script: `ml/analysis/shap_analysis.py` (same logic as `notebooks/SHAP_v2.ipynb`, writes files).
Mental health + features, 40 test tweets (10 each TP/TN/FP/FN), text held fixed for the features.

- Features move P(positive) by a median of 0.05 in total per tweet (90th percentile 0.10): a small
  adjustment on top of the text, consistent with the metrics.
- Ranking by mean |SHAP|: `emotion_volatility_10tw` (0.027, higher -> positive), `len_z` (0.011,
  longer-than-usual tweets -> negative), `self_focus` (0.005, + ), `polarity_shift` (0.005, +);
  the rest are below 0.003. The two user-relative signals (volatility over the last 10 tweets, length
  vs the user's own baseline) dominate, which is the personalisation argument of the paper.
- The fusion gate lets about 61% of the feature path through on every tweet (std 0.015); it does not
  switch the features off for some tweets and on for others.

## Cross-platform: eRisk Reddit (depression collection)

- **Data:**
  - Users with at least 50 posts: 64 depressed, 622 control.
  - 324,684 posts after each user's baseline.
  - Positive = depressed user.
  - No images, so the model text is "post + [IMG_CAP]", the same as a tweet without an image.
- **Method:**
  - No retraining, and nothing fit or tuned on Reddit. The features come from the same code as for
    Twitter, scaled with each model's Twitter train statistics.
  - Thresholds are the ones tuned on Twitter validation.
  - All users are used; the data is not subsampled.
- **Scripts:** `ml/preprocessing/process_erisk_v2.py`, `ml/evaluation/evaluate_erisk_v2.py`.

| Model trained for | Features | Reddit post AUC [95% CI] | Reddit user AUC [95% CI] | Reddit user macro F1 | User recall | Users flagged |
|---|---|---|---|---|---|---|
| Depression | text only | 0.684 [0.658, 0.710] | 0.846 [0.806, 0.886] | 0.535 | 0.92 | 44% |
| Depression | + features | 0.711 [0.683, 0.737] | 0.843 [0.805, 0.882] | 0.551 | 0.89 | 41% |
| Mental health | text only | 0.692 [0.662, 0.722] | 0.842 [0.795, 0.887] | 0.458 | 0.95 | 56% |
| Mental health | + features | 0.715 [0.685, 0.746] | 0.851 [0.813, 0.890] | 0.444 | 0.98 | 59% |

Paired bootstrap on Reddit (features minus text only):

| Model | Post AUC difference | p | User AUC difference | p |
|---|---|---|---|---|
| Mental health | +0.023 [+0.016, +0.030] | < 0.001 | +0.009 [−0.011, +0.031] | 0.39 |
| Depression | +0.027 [+0.018, +0.035] | < 0.001 | −0.004 [−0.021, +0.013] | 0.73 |

Observations:
- **The models transfer to Reddit.** Depressed users are ranked well above controls (user AUC 0.84–0.85),
  better than the depression model does on the Twitter test set (0.67). Possible reasons, not tested:
  - Reddit users have many more posts each (about 470 after the baseline), so the per-user average is
    more reliable.
  - eRisk's depressed users often write about depression directly.
- **The depression model now transfers as well as the screener** (0.846 vs 0.842; on the buggy data
  it was 0.81 vs 0.85).
- **The features are the one place they clearly help: Reddit posts.** Post AUC rises by +0.023 and
  +0.027 (p < 0.001 for both models). The gain disappears at the user level, where averaging ~470
  posts already removes most per-post noise. A possible paper point: behavioural features add
  robustness per post when the text domain changes, but not once many posts are pooled.
- **Twitter-tuned thresholds don't transfer.**
  - The models flag 41–59% of Reddit users, but only 9% are depressed, so precision is low and
    macro F1 is 0.44–0.55.
  - Report AUC as the main cross-platform result, and calibration as a limitation.
- **Not comparable with the v1 eRisk numbers.** v1 used a hand-picked 0.95 threshold on a balanced
  sample of posts, fit its feature scaling on eRisk itself, and had the broken `len_z`.

## Still to do

Order of work. Each step's command is listed so any session can pick it up.

1. [x] **Rebuild the data** with the retweet fix: `python ml/preprocessing/merge_blip.py`.
   Done 2026-11-14 23:27; counts in Setup above. No duplicate tweets, every user is one timeline.
2. [x] **Feature ablation on the new data:** `python ml/analysis/feature_ablation.py --task <task>`.
   Kept length+sentiment+language: at the user level it matches or beats all 20 features on every task
   (val user AUC, 10 vs 20: mental health 0.789 vs 0.782, depression 0.639 vs 0.592 (both means of
   seeds 42/43), anxiety 0.609 vs 0.614, bipolar 0.694 vs 0.671). Per tweet the 20-feature set is better
   for mental health (0.718 vs 0.684): on clean data the activity group now carries signal.
   Language (mostly `self_focus`) is still the strongest group everywhere.
3. [x] **Training queue:** `python ml/run_queue.py --phases 1 5 4 3`. Ran 2026-11-14 23:35 to
   2026-11-15 20:01, 14 runs, no failures (log: `out/v2/queue.log`). Weights (model.pt) kept only for
   mental health, depression and anxiety seed 42. `python ml/evaluation/summarize.py` rebuilds
   `results/summary.md`.
4. [x] **Reddit cross-test:** `python ml/evaluation/evaluate_erisk_v2.py`. Done 2026-11-15 22:18.
5. [x] **Paired bootstrap:** `python ml/evaluation/paired_bootstrap.py` (+ `--seed 43`, `--seed 44`).
   Done 2026-11-15 22:25.
6. [x] **Update this file** with the new numbers. Done 2026-11-15.
7. [x] **SHAP** on the mental health + features model: `python ml/analysis/shap_analysis.py`
   (done 2026-11-15; `--run depression__length+sentiment+language__s42` or the anxiety run also work,
   their weights are saved). `notebooks/SHAP_v2.ipynb` is the interactive version.
8. [x] **Flowchart** (`ml/analysis/flowchart.py` -> `results/figures/pipeline_flowchart.png`): user chose
   no cascade; the first parallel box is now labelled "Screener: mental health vs control".
8b. [x] **Detailed model comparison:** `python ml/analysis/compare_models.py` (done 2026-11-15).
9. [~] **Paper:** draft with all v2 results written to `paper/PAMHeD_v2.tex` (2026-11-16); figures are
   referenced from `results/figures/` via `\graphicspath`. User to review and compile.
   - Table 1: the new counts (Setup above).
   - Table 2: the 4-class results above.
   - Results: the binary table, paired bootstrap and eRisk sections above. The honest headline is that
     text drives performance; the behavioural features do not significantly help on Twitter, but do
     improve per-post ranking on Reddit.
   - Methods / lessons learned: describe the retweet issue and its fix. v1 numbers are no longer a
     valid comparison (trained on the fake users), so present v2 as the results.
10. [ ] Optional: user-level depression model (pool a user's tweets inside the model); a lower learning
    rate or 1–2 epochs, since most runs peaked at epoch 1.
