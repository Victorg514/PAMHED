"""
merge_controls.py
Copy timelines from candidate_controls.json into controls_stub.json
if the stub entry is still empty.
"""

import json, os

STUB_FILE      = "controls_stubA.json"      # has empty lists right now
TIMELINES_FILE = "candidate_controls.json"  # full timelines you scraped
OUT_FILE       = "controls_stubA.json"      # overwrite the stub

# --- load both -----------------------------------------------------------
if not (os.path.exists(STUB_FILE) and os.path.exists(TIMELINES_FILE)):
    raise FileNotFoundError("Make sure both JSON files are in this directory.")

stub      = json.load(open(STUB_FILE, encoding="utf-8"))
timelines = json.load(open(TIMELINES_FILE, encoding="utf-8"))

# --- merge ---------------------------------------------------------------
updated = 0
missing = 0

for user in stub:
    if stub[user]:                       # already filled – leave as-is
        continue
    if user in timelines and timelines[user]:
        stub[user] = timelines[user]     # copy full timeline in
        updated += 1
    else:
        missing += 1                     # no data found for this user

# --- save ----------------------------------------------------------------
json.dump(stub, open(OUT_FILE, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print(f"Finished: {updated} users updated, {missing} still empty.")
print(f"Result written back to {OUT_FILE}")
