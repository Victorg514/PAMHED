import json, os, pathlib, sys

ANX = "data/controls_stubA.json"
OUT = "data/used_controls.txt"

used = set(json.load(open(ANX, encoding="utf-8")).keys())
pathlib.Path(OUT).write_text("\n".join(sorted(used)), encoding="utf-8")
print(f"{len(used)} controls recorded → {OUT}")
