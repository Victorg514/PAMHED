import os, sys, json, re, requests, pathlib
from ntscraper import Nitter
import ntscraper.nitter as ntr
from bs4 import BeautifulSoup

# ─── monkey-patch avatar bug guard you already use ─────────────────────────
_orig_get_user = ntr.Nitter._get_user
def _safe_get_user(self, tweet, is_encrypted):
    try:
        return _orig_get_user(self, tweet, is_encrypted)
    except IndexError:
        uname = tweet.find("a", class_="username")
        fname = tweet.find("a", class_="fullname")
        return {
            "id": None,
            "username": uname.text.lstrip("@") if uname else "unknown",
            "fullname": fname.text if fname else "unknown",
            "avatar_url": None,
        }
ntr.Nitter._get_user = _safe_get_user
# ───────────────────────────────────────────────────────────────────────────

if len(sys.argv) != 2:
    print("Give exactly one username (without @).")
    sys.exit(1)

user = sys.argv[1].lstrip("@")
out_dir = pathlib.Path("data/images") / user
out_dir.mkdir(parents=True, exist_ok=True)

print(f"▸ Scraping timeline for @{user}")
scraper  = Nitter(log_level=1, skip_instance_check=True,
                  instance="http://localhost:8080")

timeline = scraper.get_tweets(user, "user", -1)["tweets"]
print(f"  • {len(timeline)} tweets fetched")

# --- save tweets ----------------------------------------------------------
json_path = pathlib.Path("data") / f"{user}.json"
json_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2),
                     encoding="utf-8")
print(f"  • timeline written to {json_path}")

# --- collect & download images -------------------------------------------
img_urls = []
for tw in timeline:
    # ntscraper stores photo URLs under key 'photos' (list) if present
    for url in tw.get("photos", []):
        if url not in img_urls:
            img_urls.append(url)

print(f"  • {len(img_urls)} unique images to download")

sess = requests.Session()
for idx, url in enumerate(img_urls, 1):
    ext  = pathlib.Path(url).suffix.split("?")[0] or ".jpg"
    fname = out_dir / f"{idx:04d}{ext}"
    try:
        with sess.get(url, timeout=30, stream=True) as r:
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
    except Exception as e:
        print(f"    ✗ failed {url} → {e}")
        continue
    print(f"    ✓ {fname.name}")

print("\nDone.")

