# match_controls.py
import json, csv, math, datetime as dt
COND_TIMELINES = "data/all_conditions_timelines.json"
CAND_JSON = "candidate_controls.json"
OUT_PAIR  = "matched_pairs.json"


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



def parse_date(raw: str) -> dt.datetime:
    """
    Convert Nitter's 'Dec 13, 2024 · 10:46 AM UTC'
    to a datetime object (naïve, assumed UTC).
    """
    cleaned = (
        raw.replace("·", "")     # drop the bullet
           .replace("UTC", "")   # drop timezone label
           .strip()
    )                            # 'Dec 13, 2024 10:46 AM'
    return dt.datetime.strptime(cleaned, "%b %d, %Y %I:%M %p")

# compute metrics for condition users
cond_data = json.load(open(COND_TIMELINES, encoding="utf-8"))
cond_metrics = {u: metrics(tl) for u, tl in cond_data.items()}

# compute metrics for every control candidate
cand_data = json.load(open(CAND_JSON, encoding="utf-8"))
cand_metrics = {}
for u, tl in cand_data.items():
    if not tl: continue
    n = len(tl)
    ds = [parse_date(t["date"]) for t in tl]
    rate = round(n / ((max(ds) - min(ds)).days or 1), 2)
    cand_metrics[u] = (n, rate)

# greedy nearest neighbour (each control used once)
pairs, used = {}, set()
for cu, cn, cr in cond_metrics:
    best, score = None, 1e9
    for vu, (vn, vr) in cand_metrics.items():
        if vu in used: continue
        d = math.hypot(vn - cn, vr - cr)
        if d < score:
            best, score = vu, d
    pairs[cu] = best
    used.add(best)

json.dump(pairs, open(OUT_PAIR, "w"), indent=2)
print(f"Matched {len(pairs)} condition users → {OUT_PAIR}")
