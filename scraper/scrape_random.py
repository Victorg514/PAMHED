import json, random
from ntscraper import Nitter
import ntscraper.nitter as ntr

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
ntr.Nitter._get_user = _safe_get_user

OUT_JSON = "candidate_controls2.json"

scraper  = Nitter(log_level=1, skip_instance_check=True,)

keywords = ["and", "like", "cat", "sun", "happy", "my"]
pool = {}

for kw in keywords:
    hits = scraper.get_tweets(kw, "term", 100, instance="http://localhost:8080",
                              language="en")["tweets"]
    for t in hits:
        u = t["user"]["username"].lstrip("@")
        pool.setdefault(u, [])     # timeline will be filled later

json.dump(pool, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
print(f"Collected {len(pool)} candidate controls → {OUT_JSON}")
