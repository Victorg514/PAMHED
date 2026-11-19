# Personalized Mental Health Detection from Social Media Behavior (PAMHeD)

Detects depression, anxiety and bipolar disorder from Twitter/X timelines. The hypothesis is that deviation from a user's own behavioral baseline adds signal to the post text. Each task is a binary classifier built on MentalRoBERTa, trained with and without 10 behavioral deviation features (a text-only baseline and a "delta" model with the same architecture), plus a 4-class model for comparison.

- Paper: ([PAMHeD.pdf](<PAMHeD.pdf>), [last progress report](<396 Progress report 3 (2).pdf>))

## Structure

```
scraper/            Nitter-based Twitter/X scrapers
ml/preprocessing/   timeline merging, control matching, feature engineering, eRisk parsing
ml/features.py      feature groups, train-only scaling, task labels
ml/training/        train.py: one script for every task and feature set
ml/run_queue.py     runs the full set of training jobs and writes the summary
ml/evaluation/      metrics, summary table, paired bootstrap, eRisk cross-platform test
ml/analysis/        feature ablation, model comparison, SHAP, pipeline flowchart
ml/utils/           image download, BLIP captioning, filtering helpers
ml/testing/         live single-user prediction pipeline, sanity checks (v1)
ml/legacy/          v1 training and evaluation scripts (kept for reference, not used)
notebooks/          SHAP_v2.ipynb (interactive explainability); legacy/ holds the v1 notebooks
results/            tables (.md), figures/, and the notes above
nitter/, tools/     Nitter source/config and session generators
```

Outputs go to `out/v2/<task>__<features>__s<seed>/` (predictions, metrics, config; weights for a few runs). `out/` and `data/` are not tracked.

## Setup

Run all commands from the repo root.

```bash
pip install -r requirements.txt scipy langdetect shap
python -c "import nltk; nltk.download('vader_lexicon')"
huggingface-cli login   # mental/mental-roberta-base is gated; request access first
mkdir data
```

## 1. Get the data (choose one)

### Option A: Use the collected dataset

Download **[link to dataset]** into `data/`. If it includes `final.parquet`, skip to step 3. Otherwise run step 2.

### Option B: Scrape a new dataset with Nitter

1. **Start Nitter:**
   ```bash
   python tools/create_session_browser.py <user> <pass> [totp] --append sessions.jsonl
   cp nitter/nitter.conf nitter.conf
   docker compose up -d   # serves http://localhost:8080
   ```
2. **Scrape condition users.** In `scraper/main.py`, set `SEARCH_TERM` and the output files for each condition, and raise the count limits. Run it once per condition, then fill out each user's timeline with `update_timelines.py` (set its `OUT_FILE` to the matching file).
   - depression → `data/user_timelines.json`
   - anxiety → `data/user_timelinesA.json`
   - bipolar → `data/user_timelinesB.json`
   ```bash
   python scraper/main.py
   python ml/preprocessing/update_timelines.py
   ```
3. **Scrape and match controls:**
   ```bash
   python scraper/scrape_random.py                 # set OUT_JSON = "candidate_controls.json"
   python ml/preprocessing/update_timelines.py     # set OUT_FILE = "candidate_controls.json"
   python ml/utils/filter.py
   python ml/preprocessing/merge_conditions.py
   python -c "open('data/controls_stubA.json','w').write('{}')"
   python ml/utils/known_controls.py
   python ml/preprocessing/match_missing_controls.py
   python ml/preprocessing/merge_all_controls.py   # → data/all_controls_timelines.json
   ```
4. **Caption images (optional).** Run `download_images.py` once per timeline file (set `TIMELINE_FILE` each time), then `blip_base.py` once:
   ```bash
   python ml/utils/download_images.py
   python ml/utils/blip_base.py
   ```

## 2. Build features

```bash
python ml/preprocessing/merge_blip.py   # → data/final.parquet
```

Assigns tweets to the timeline owner, drops retweets and duplicates, filters non-English users, builds each user's baseline (first 40% of the timeline) and the deviation features, and writes the user-level train/val/test split. Features are saved unscaled; scaling is fit on the train split at training time.

## 3. Train

```bash
# one run: --task mental_health | depression | anxiety | bipolar | 4class
python ml/training/train.py --task depression --features none                        # text only
python ml/training/train.py --task depression --features length,sentiment,language   # + 10 features

# everything used in the paper (14 runs, ~22 h on an RTX 3060), then the summary table
python ml/run_queue.py --phases 1 5 4 3
```

Finished runs are skipped, so the queue can be stopped and restarted. `python ml/evaluation/summarize.py` rebuilds `results/summary.md` from the saved metrics.

## 4. Evaluate

```bash
python ml/evaluation/paired_bootstrap.py                  # text only vs + features, paired over test users
python ml/evaluation/paired_bootstrap.py --seed 43         # (anxiety was also run with seeds 43 and 44)
```

**Cross-platform (optional).** Get the eRisk depression collection from [erisk.irlab.org](https://erisk.irlab.org/) (requires a signed user agreement) and put it in `data/reddit_test_depression/`. Then:

```bash
python ml/preprocessing/compile-erisk_chunks.py
python ml/preprocessing/parse_erisk_xml.py          # → data/erisk_parsed_raw.parquet
python ml/preprocessing/process_erisk_v2.py         # → data/erisk_v2.parquet
python ml/evaluation/evaluate_erisk_v2.py           # Twitter models and thresholds, nothing fit on Reddit
```

## 5. Analysis

```bash
python ml/analysis/feature_ablation.py --task mental_health   # which feature groups help (features only, CPU)
python ml/analysis/compare_models.py                          # confusion, calibration, when the features help
python ml/analysis/shap_analysis.py                           # SHAP on a saved run (GPU); or notebooks/SHAP_v2.ipynb
python ml/analysis/flowchart.py                               # pipeline figure
```

All outputs land in `results/` (`summary.md`, `paired_bootstrap_s*.md`, `erisk_cross_platform.md`, `model_comparison.md`, `shap_*.md`, `figures/`).
