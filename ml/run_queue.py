"""
Runs all v2 training jobs one after another (e.g. overnight), then writes the summary table.

  python ml/run_queue.py --phases 1 5 4 3   # the full plan used in the paper, ~24 h
  python ml/run_queue.py                     # phase 1 only: mental health + depression, text only vs features
  python ml/run_queue.py --dry_run           # show what would run

Phases: 1 = mental health + depression, 5 = anxiety + bipolar, 4 = 4-class,
3 = anxiety seeds 43 and 44, 2 = all 20 features (optional).

Finished runs (metrics.json present) are skipped, so the queue can be stopped and restarted.
Only the runs in SAVE_WEIGHTS keep their weights (~500 MB each), to save disk space.
"""
import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "out" / "v2"

# Feature set for the "with features" model, chosen by the CPU ablation on the validation split
# (ml/analysis/feature_ablation.py): at the user level it beats all 20 features on three of four tasks
# and is within 0.005 on anxiety
FEATURES = "length,sentiment,language"
TASKS = ("mental_health", "depression")
# Only these keep model.pt (~500 MB each): needed later for the eRisk test and SHAP. Every run keeps its predictions.
SAVE_WEIGHTS = {("mental_health", 42), ("depression", 42), ("anxiety", 42)}

PHASES = {
    1: [(task, feats, 42) for task in TASKS for feats in ("none", FEATURES)],               # core comparison
    2: [(task, "default", 42) for task in TASKS],                                           # all 20 features
    3: [("anxiety", feats, seed) for seed in (43, 44) for feats in ("none", FEATURES)],  # seeds, for the one task with a feature gain
    4: [("4class", feats, 42) for feats in ("none", FEATURES)],
    5: [(task, feats, 42) for task in ("anxiety", "bipolar") for feats in ("none", FEATURES)],  # other conditions
}


def run_name(task, feats, seed):
    return f"{task}__{feats.replace(',', '+')}__s{seed}"


def main(args):
    jobs = [job for phase in args.phases for job in PHASES[phase]]
    log_path = RUNS_DIR / "queue.log"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for i, (task, feats, seed) in enumerate(jobs, 1):
        name = run_name(task, feats, seed)
        if (RUNS_DIR / name / "metrics.json").exists():
            print(f"[{i}/{len(jobs)}] skip {name} (done)")
            continue
        cmd = [sys.executable, str(ROOT / "ml" / "training" / "train.py"), "--task", task, "--features", feats,
               "--seed", str(seed)] + ([] if (task, seed) in SAVE_WEIGHTS else ["--no-save_model"]) + args.extra
        print(f"[{i}/{len(jobs)}] {' '.join(cmd[1:])}")
        if args.dry_run:
            continue
        t0 = time.time()
        result = subprocess.run(cmd, cwd=ROOT)
        status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')}  {name}  {status}  {(time.time() - t0) / 60:.0f} min\n")
        if result.returncode != 0 and not args.keep_going:
            sys.exit(f"Stopped: {name} failed. See output above; rerun the queue to resume.")

    if not args.dry_run:
        subprocess.run([sys.executable, str(ROOT / "ml" / "evaluation" / "summarize.py")], cwd=ROOT)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phases", type=int, nargs="+", default=[1], choices=sorted(PHASES))
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--keep_going", action="store_true", help="Continue with the next job if one fails")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to train.py, after '--'")
    args = p.parse_args()
    args.extra = [a for a in args.extra if a != "--"]
    main(args)
