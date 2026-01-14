
import json, os, itertools

INPUTS = [
    "data/user_timelines.json",
    "data/user_timelinesB.json",
    "data/user_timelinesA.json",
]
OUT    = "data/all_conditions_timelines.json"

combined = {}
for path in INPUTS:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    combined.update(json.load(open(path, encoding="utf-8")))

json.dump(combined, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"{len(combined)} total condition users → {OUT}")
