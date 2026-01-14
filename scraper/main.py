import json
import os
from time import sleep
from ntscraper import Nitter
import re
from bs4 import BeautifulSoup   # ntscraper already depends on it
import ntscraper.nitter as ntr
from datetime import date, timedelta

_orig_get_user = ntr.Nitter._get_user

def _safe_get_user(self, tweet, is_encrypted):
    """Fallback when avatar path is missing."""
    try:
        return _orig_get_user(self, tweet, is_encrypted)
    except IndexError:
        # Minimal fallback: just grab username & name from anchors
        uname = tweet.find("a", class_="username")
        fname = tweet.find("a", class_="fullname")
        return {
            "id": None,
            "username": uname.text.lstrip("@") if uname else "unknown",
            "fullname": fname.text if fname else "unknown",
            "avatar_url": None,
        }
    

# Monkey-patch the class method
ntr.Nitter._get_user = _safe_get_user

# ─── CONFIG ────────────────────────────────────────────────────────────────
INSTANCE_URL       = "http://localhost:8080"
SEARCH_TERM        = "Mahomes"
WINDOW_DAYS         = 30                 # size of each backward slice
STOP_AT             = date(2024, 1, 1)
SEARCH_COUNT       = 1                 # tweets pulled for the term search
MAX_USERS_TO_SCRAPE = 1                 # e.g. 3 users.  None → all new users
TIMELINE_DEPTH     = 1                # tweets per user timeline
TERM_OUT_FILE      = "data/bipolar_tweets.json"
USER_OUT_FILE      = "data/user_timelinesB.json"
PAUSE_BETWEEN_USERS = 2                 # seconds
LANGUAGE_FILTER     = "en" 
# ───────────────────────────────────────────────────────────────────────────

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or default
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def tid(tweet):
    return tweet["link"].split("/")[-1]

def main():
    scraper = Nitter(log_level=1, skip_instance_check=True)
    print(f"Using Nitter instance {INSTANCE_URL}\n")

    # ── 1) TERM SEARCH ────────────────────────────────────────────────────
    term_tweets = []
    until_date = date(2025,7,30)
    remaining   = SEARCH_COUNT

    while until_date >= STOP_AT and remaining > 0:
        # next window, but never let since_date drop before STOP_AT
        since_date = max(until_date - timedelta(days=WINDOW_DAYS - 1), STOP_AT)

        # ask only for the tweets we still need
        fetch_n = min(remaining, 500)     # 500 is ntscraper’s max per call
        print(f"Slice {since_date} → {until_date}  (need {remaining})")

        slice_res = scraper.get_tweets(
            SEARCH_TERM,
            "term",
            fetch_n,
            instance=INSTANCE_URL,
            language=LANGUAGE_FILTER,
            since=str(since_date),
            until=str(until_date),
        )
        slice_tweets = slice_res.get("tweets", [])
        print(f"  • got {len(slice_tweets)}")

        term_tweets.extend(slice_tweets)
        remaining -= len(slice_tweets)

        until_date = since_date - timedelta(days=1)

    print(f"\nTotal fetched across slices: {len(term_tweets)} tweets")

    # ── 2) LOAD & DEDUP depression_tweets.json ────────────────────────────
    store = load_json(TERM_OUT_FILE, {"tweets": []})
    seen_ids = {tid(t) for t in store["tweets"]}

    new_term_tweets = [t for t in term_tweets if tid(t) not in seen_ids]
    print(f"  ↳ {len(new_term_tweets)} of them are NEW.\n")

    if new_term_tweets:
        store["tweets"].extend(new_term_tweets)
        save_json(TERM_OUT_FILE, store)
        print(f"Appended new tweets to {TERM_OUT_FILE}\n")

    # ── 3) PICK USERS ONLY FROM *NEW* TWEETS ──────────────────────────────
    usernames = list({t["user"]["username"].lstrip("@") for t in new_term_tweets})
    if MAX_USERS_TO_SCRAPE is not None:
        usernames = usernames[:MAX_USERS_TO_SCRAPE]

    print(f"Will scrape timelines for {len(usernames)} user(s): {usernames}\n")
    if not usernames:
        print("Nothing to do – no new users found.")
        return

    # ── 4) LOAD EXISTING USER TIMELINES / BUILD SEEN SETS ────────────────
    user_data = load_json(USER_OUT_FILE, {})
    seen_ids_per_user = {
        u: {tid(t) for t in tweets} for u, tweets in user_data.items()
    }

    # ── 5) SCRAPE EACH USER TIMELINE ─────────────────────────────────────
    for user in usernames:
        print(f"↳ Scraping {TIMELINE_DEPTH} tweets for @{user}")
        try:
            timeline = scraper.get_tweets(
                user, mode="user", number=TIMELINE_DEPTH, instance=INSTANCE_URL
            ).get("tweets", [])

            user_data.setdefault(user, [])
            seen_ids_per_user.setdefault(user, set())

            fresh = [t for t in timeline if tid(t) not in seen_ids_per_user[user]]
            if fresh:
                user_data[user].extend(fresh)
                seen_ids_per_user[user].update(tid(t) for t in fresh)
                print(f"   • added {len(fresh)} new tweets")
            else:
                print("   • no new tweets")
        except Exception as e:
            print(f"   • failed: {e}")
        sleep(PAUSE_BETWEEN_USERS)

    # ── 6) SAVE MERGED TIMELINES ─────────────────────────────────────────
    save_json(USER_OUT_FILE, user_data)
    print(f"\nFinished. Data stored in {USER_OUT_FILE}")

if __name__ == "__main__":
    main()
