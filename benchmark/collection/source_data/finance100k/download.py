"""Save up to 5000 images from Sujet-Finance-QA-Vision-100k into ./data/ (next to this script)."""

import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

DATASET = "sujet-ai/Sujet-Finance-QA-Vision-100k"
OUTDIR = Path(__file__).resolve().parent / "data"
MAX_IMAGES = 5000


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    ds = load_dataset(DATASET)
    merged = concatenate_datasets([ds["train"], ds["test"]])
    OUTDIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, row in enumerate(merged):
        if saved >= MAX_IMAGES:
            break
        img = row["image"]
        ext = (img.format or "png").lower()
        name = Path(str(row.get("doc_id", "noid"))).stem  # drop any extension in the doc id
        if ext in ("jpg", "jpeg") and img.mode != "RGB":
            img = img.convert("RGB")
        img.save(OUTDIR / f"{i:06d}_{name}.{ext}")
        saved += 1
    print(f"Saved {saved} images to {OUTDIR}")


if __name__ == "__main__":
    main()
