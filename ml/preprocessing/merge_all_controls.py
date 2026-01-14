import json, pathlib

ANX  = "data/controls_stubA.json"
NEW  = "data/new_controls.json"
OUT  = "data/all_controls_timelines.json"

merged = {}
for path in (ANX, NEW):
    merged.update(json.load(open(path, encoding="utf-8")))

json.dump(merged, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"{len(merged)} total controls → {OUT}")
