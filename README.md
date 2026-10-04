# Personalized Mental Health Detection from Social Media Behavior

Detects depression, anxiety, and bipolar disorder from Twitter/X timelines. The hypothesis is that deviation from a user's own behavioral baseline is a stronger signal than post content alone. The repo includes a 4-class model and a One-vs-Rest ensemble of binary classifiers, both built on MentalRoBERTa. Refer to the last progress report and final paper for details: [396 Progress report 3 (2).pdf](<396 Progress report 3 (2).pdf>) , [PAMHeD.pdf](<PAMHeD.pdf>) .

## Structure

```
scraper/            Nitter-based Twitter/X scrapers
ml/preprocessing/   timeline merging, control matching, feature engineering, eRisk parsing
ml/training/        baseline (text-only) and delta (text + behavioral deviation) models
ml/evaluation/      test-set and cross-platform (eRisk) evaluation
ml/utils/           image download, BLIP captioning, filtering helpers
ml/testing/         live single-user prediction pipeline, sanity checks
notebooks/          data visualization and SHAP explainability
nitter/, tools/     Nitter source/config and session generators
```

## Setup

Run all commands from the repo root.

```bash
pip install -r requirements.txt scipy langdetect
python -c "import nltk; nltk.download('vader_lexicon')"
huggingface-cli login   # mental/mental-roberta-base is gated; request access first
mkdir data
```

## 1. Get the data (choose one)

### Option A: Use the collected dataset

Download **[link to dataset]** into `data/`. If it includes `final.parquet` and `class_weights.npy`, skip to step 3. Otherwise run step 2.

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
python ml/preprocessing/merge_blip.py   # → data/final.parquet, data/class_weights.npy
```

## 3. Train

```bash
python ml/training/train_baseline.py    # 4-class
python ml/training/train_delta.py
# binary; --positive_class: mental_health | depression | anxiety | bipolar
python ml/training/train_binary_baseline_manual.py --positive_class depression
python ml/training/train_binary_delta_manual.py    --positive_class depression
```

## 4. Evaluate

```bash
python ml/evaluation/evaluate_models.py
python ml/evaluation/evaluate_binary_models.py --positive_class depression   # delete cache/ after retraining
```

**Cross-platform (optional).** Get the eRisk depression collection from [erisk.irlab.org](https://erisk.irlab.org/) (requires a signed user agreement) and put it in `data/reddit_test_depression/`. Then:

```bash
python ml/preprocessing/compile-erisk_chunks.py
python ml/preprocessing/parse_erisk_xml.py
python ml/preprocessing/process_erisk_data.py
python ml/evaluation/evaluate_erisk_balanced_delta.py --positive_class depression
```

**Explainability.** Use `notebooks/SHAP.ipynb` and `notebooks/data_visual.ipynb`.
