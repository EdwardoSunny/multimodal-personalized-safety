"""Save up to 1000 meme images from Multi3Hate into ./data/ (next to this script)."""

import argparse
import re
from pathlib import Path

from datasets import load_dataset

DATASET = "MinhDucBui/Multi3Hate"
OUTDIR = Path(__file__).resolve().parent / "data"
MAX_IMAGES = 1000


def slug(s, fallback="noid"):
    return re.sub(r"_+", "_", re.sub(r"[^\w\-]+", "_", str(s or fallback))).strip("_") or fallback


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    ds = load_dataset(DATASET)["train"]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, row in enumerate(ds):
        if saved >= MAX_IMAGES:
            break
        img = row["image"]
        ext = (img.format or "png").lower().replace("jpeg", "jpg")
        if ext == "jpg" and img.mode != "RGB":
            img = img.convert("RGB")
        name = f'{i:06d}_{slug(row.get("Meme ID"))}_{slug(row.get("Language"))}_{slug(row.get("Template Name"))}.{ext}'
        try:
            img.save(OUTDIR / name)
            saved += 1
        except Exception as e:
            print(f"skip {i}: {e}")

    print(f"Saved {saved} images to {OUTDIR}")


if __name__ == "__main__":
    main()
