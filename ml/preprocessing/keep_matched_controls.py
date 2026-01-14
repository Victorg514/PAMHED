import json

# --- Load files ---
with open("candidate_controls.json", "r", encoding="utf-8") as f:
    tweets_data = json.load(f)

with open("matched_pairs.json", "r", encoding="utf-8") as f:
    user_pairs = json.load(f)

# --- Extract the set of usernames to keep (from the VALUES in user_pairs) ---
users_to_keep = set(user_pairs.values())

# --- Filter tweets_data ---
filtered_data = {
    user: tweets
    for user, tweets in tweets_data.items()
    if user in users_to_keep
}

# --- Save filtered result ---
with open("all_controls_timelines.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"Kept {len(filtered_data)} users")
