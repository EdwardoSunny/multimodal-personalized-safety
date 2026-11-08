"""Download MPS-Bench from the Hugging Face Hub into benchmark/data/.

The dataset is gated: accept the terms on the dataset page, then authenticate
with `hf auth login` (or set HF_TOKEN) before running this script.

After downloading, every image referenced by benchmark_data.json is checked; if any is
missing (the Hub's rate limit can cut a download short), the script lists them and exits
with status 1. Rerunning it fetches only the missing files.
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "EdwardoSunny/multimodal-personalized-safety"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "benchmark" / "data"
SHOW_MISSING = 5


def missing_files(dest: Path) -> list[str]:
    """Files that benchmark_data.json (or the download itself) needs but that are not in `dest`."""
    missing = [name for name in ("benchmark_data.json", "image_sources.json") if not (dest / name).is_file()]
    if "benchmark_data.json" in missing:
        return missing
    with open(dest / "benchmark_data.json") as f:
        records = json.load(f)
    # image_path is relative to benchmark/ ("data/images/<id>.<ext>"); dest plays the role of benchmark/data
    for record in records:
        image = dest / "images" / Path(record["image_path"]).name
        if not image.is_file():
            missing.append(str(image.relative_to(dest)))
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Output directory (default: benchmark/data)")
    parser.add_argument("--revision", default=None, help="Dataset revision (branch, tag, or commit)")
    args = parser.parse_args()

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.dest,
        allow_patterns=["benchmark_data.json", "image_sources.json", "images/*"],
    )
    missing = missing_files(args.dest)
    if missing:
        sys.exit(
            f"Error: the download is incomplete: {len(missing)} file(s) missing from {args.dest}, e.g. "
            f"{', '.join(missing[:SHOW_MISSING])}.\n"
            "This usually means the Hugging Face Hub rate limit cut the download short. Rerun this script "
            "(already downloaded files are kept) until it reports success."
        )
    n_images = sum(1 for _ in (args.dest / "images").iterdir())
    print(f"Downloaded MPS-Bench to {args.dest} ({n_images} images; every image in benchmark_data.json is present)")


if __name__ == "__main__":
    main()
