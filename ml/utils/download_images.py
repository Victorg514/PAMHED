import json
import requests
import time
from pathlib import Path
from urllib.parse import urlparse

# ------------ CONFIG -------------------------------------------------

TIMELINE_FILE = Path("data/all_controls_timelines.json")       # your big file
IMAGE_DIR      = Path("data/images")               # where pics will live
BASE_URL       = "https://twitter.com"             # prepend if pic path starts with "/"
HEADERS        = {"User-Agent": "MH-Research-Image-Scraper"}

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
session = requests.Session()
session.headers.update(HEADERS)

# ------------ HELPER -------------------------------------------------

def download_image(url: str, out_path: Path) -> bool:
    """Download URL to out_path. Return True if saved, False if failed/existed."""
    if out_path.exists():
        return False                                  # already have it
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except Exception as e:
        print("Download failed:", url, e)
        return False

def clean_url(raw: str) -> str:
    """Add BASE_URL if needed and strip Nitter query fragments (#m, ?name=orig, …)."""
    url = raw
    if raw.startswith("/"):                           # relative path from Nitter
        url = BASE_URL + raw
    url = url.split("#")[0].split("?")[0]             # remove fragments/queries
    return url

# ------------ MAIN ---------------------------------------------------

print("Loading timeline JSON …")
with TIMELINE_FILE.open("r", encoding="utf-8") as f:
    timelines = json.load(f)                          # keys = usernames, values = list[tweets]

total, saved = 0, 0
for tweets in timelines.values():
    for tw in tweets:
        tid = tw["id"]
        pics = tw.get("pictures", [])                 # List[str]
        for idx, raw_url in enumerate(pics):
            url = clean_url(raw_url)
            ext = Path(urlparse(url).path).suffix.lstrip(".") or "jpg"
            fname = f"{tid}_{chr(97+idx)}.{ext}"      # 97 = 'a', so _a, _b, …
            if download_image(url, IMAGE_DIR / fname):
                saved += 1
            total += 1
        time.sleep(0.05)                              # gentle on CDN (20 req/s)

print(f"Finished. Tried {total} images, saved {saved} new files into {IMAGE_DIR}")
