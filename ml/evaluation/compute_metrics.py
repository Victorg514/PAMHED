# compute_metrics.py
import json, csv, datetime as dt, os, statistics

IN_JSON  = "user_timelinesA.json"
OUT_CSV  = "condition_metrics.csv"

def parse_date(raw: str) -> dt.datetime:
    cleaned = raw.replace("·", "").replace("UTC", "").strip()
    return dt.datetime.strptime(cleaned, "%b %d, %Y %I:%M %p")

def tweets_per_day(tl):
    if len(tl) == 0:
        return 0.0                       # ← guard: empty timeline
    dates = [parse_date(t["date"]) for t in tl]
    span  = (max(dates) - min(dates)).days or 1
    return round(len(tl) / span, 2)

def main():
    data = json.load(open(IN_JSON, encoding="utf-8"))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["user", "n_tweets", "rate"])
        for user, tl in data.items():
            if not tl:                    # skip or include with zeros
                continue                  # or: wr.writerow([user, 0, 0])
            wr.writerow([user, len(tl), tweets_per_day(tl)])

    print(f"→ {OUT_CSV}")

if __name__ == "__main__":
    main()