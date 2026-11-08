"""Download post images from Reddit post dumps.

Reads every subreddit_data/r_<subreddit>_posts.jsonl (one Reddit post JSON object per line,
next to this script) and saves up to MAX_IMAGES_PER_SUBREDDIT images per subreddit to
images/<subreddit>/<post id>.<ext>. gather_images.py treats each images/<subreddit>/ folder
as its own source.
"""

import argparse
import json
import os
from pathlib import Path

import requests
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
POSTS_DIR = HERE / "subreddit_data"
IMAGES_DIR = HERE / "images"
MAX_IMAGES_PER_SUBREDDIT = 2000
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp")


def best_image_url(post):
    url = post.get("url", "")
    if url.lower().endswith(IMAGE_EXTS):
        return url
    if "preview" in post and post["preview"].get("images"):
        return post["preview"]["images"][0]["source"]["url"].replace("&amp;", "&")
    return None


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = "reddit-img-scraper"
    total_saved = 0
    totals = {"malformed_json": 0, "no_image_url": 0, "request_error": 0, "http_error": 0}

    for fn in sorted(os.listdir(POSTS_DIR)):
        if not fn.endswith("_posts.jsonl"):
            continue

        # "r_pics_posts.jsonl" -> "pics"
        subreddit_name = fn.replace("r_", "").replace("_posts.jsonl", "")
        subreddit_folder = IMAGES_DIR / subreddit_name
        subreddit_folder.mkdir(parents=True, exist_ok=True)

        saved_for_sub = 0
        counts = dict.fromkeys(totals, 0)
        path = POSTS_DIR / fn
        with open(path, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total, desc=f"Loading {fn}", unit="lines", ncols=100):
                if saved_for_sub >= MAX_IMAGES_PER_SUBREDDIT:
                    break
                try:
                    post = json.loads(line)
                except json.JSONDecodeError:
                    counts["malformed_json"] += 1
                    continue

                url = best_image_url(post)
                if not url:
                    counts["no_image_url"] += 1
                    continue
                ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
                out = subreddit_folder / f"{post.get('id', 'post')}{ext}"
                try:
                    r = session.get(url, timeout=15)
                except requests.RequestException:
                    counts["request_error"] += 1
                    continue
                if not r.ok:
                    counts["http_error"] += 1
                    continue
                out.write_bytes(r.content)
                saved_for_sub += 1
                total_saved += 1

        print(f"r/{subreddit_name}: saved {saved_for_sub}; skipped {counts}")
        for k, v in counts.items():
            totals[k] += v

    print(f"Done. Saved {total_saved} images total (up to {MAX_IMAGES_PER_SUBREDDIT} per subreddit); skipped {totals}.")


if __name__ == "__main__":
    main()
