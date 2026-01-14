
import json, csv, math, datetime as dt, pathlib

COND_TIMELINES = "data/all_conditions_timelines.json"
USED_TXT       = "data/used_controls.txt"        # from step 2
CAND_JSON      = "final_candidate_controls.json"  # big neutral pool
ANX_CNTL_JSON  = "data/controls_stubA.json"
OUT_CNTL_JSON  = "data/new_controls.json"        # newly picked controls
OUT_PAIR_JSON  = "data/all_pairs.json"           # cond→control map

# ── load helpers (date format parsing) ───────────────────────────────────
def parse(raw):
    return dt.datetime.strptime(
        raw.replace("·", "").replace("UTC", "").strip(),
        "%b %d, %Y %I:%M %p"
    )

def metrics(tl):
    n = len(tl)
    if n == 0:
        return n, 0.0
    dates = [parse(t["date"]) for t in tl]
    span  = (max(dates) - min(dates)).days or 1
    return n, round(n / span, 2)          # count, rate
# ------------------------------------------------------------------------

cond_data = json.load(open(COND_TIMELINES, encoding="utf-8"))
used_controls = set(pathlib.Path(USED_TXT).read_text().splitlines())
anx_controls  = json.load(open(ANX_CNTL_JSON, encoding="utf-8"))

# compute metrics for condition users
cond_metrics = {u: metrics(tl) for u, tl in cond_data.items()}

# load + metric candidates
cand_data   = json.load(open(CAND_JSON, encoding="utf-8"))
cand_met    = {u: metrics(tl) for u, tl in cand_data.items() if u not in used_controls}

# greedy nearest neighbour
pairs, new_controls = {}, {}
for cu, (cn, cr) in cond_metrics.items():
    # if anxiety already matched, keep it
    if cu in anx_controls:
        pairs[cu] = cu  # points to same key in final control merge
        continue
    best, score = None, 1e9
    for vu, (vn, vr) in cand_met.items():
        if vu in used_controls:
            continue
        d = math.hypot(vn - cn, vr - cr)
        if d < score:
            best, score = vu, d
    if best is None:
        print(f"!! no control found for {cu}")
        continue
    pairs[cu] = best
    used_controls.add(best)
    new_controls[best] = cand_data[best]   # copy timeline

json.dump(new_controls, open(OUT_CNTL_JSON, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
json.dump(pairs,       open(OUT_PAIR_JSON, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print(f"Picked {len(new_controls)} NEW controls → {OUT_CNTL_JSON}")
