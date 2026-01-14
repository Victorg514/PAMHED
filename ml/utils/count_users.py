import json

# Path to your file
path = "data/user_timelinesA.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- Adjust ONE of the blocks below to match your file’s structure ----

# A) If the file is a flat list of tweets/retweets
#    and each item has something like  tweet["user"]["screen_name"]
#users = {tweet["user"]["screen_name"] for tweet in data}

# B) If each “section” is a dict keyed by a username,
#    e.g. { "alice": [...tweets...], "bob": [...tweets...] }
users = data.keys()

# C) If each section is a list of dicts, each starting with {"username": "alice", ...}
# users = {section[0]["username"] for section in data}

print("Number of unique users:", len(users))
