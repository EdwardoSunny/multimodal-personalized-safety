"""Label the overall emotional valence (positive / neutral / negative) of every image in data/images/.

Usage (from benchmark/):
    python generation/label_images.py

Requires OPENAI_API_KEY. Writes data/image_valence_labels.json, mapping image file name to
{"valence", "confidence", "explanation"}; the file is rewritten atomically after every image.

If labeling an image raises (e.g. its reply is not valid JSON, its valence is not one of
positive / neutral / negative, or the API call fails), the model is queried once more for that
image and the new reply is passed through the JSON-repair prompt (lib/vlm_utils.fix_json) before
parsing; images that still fail are listed and counted (invalid valence labels separately). A
label is only stored with a valid valence.

An existing labels file is resumed: every label in it is kept and only images without a label
are labeled; a file that cannot be parsed aborts the run untouched. --overwrite starts a fresh
file.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import IMAGE_EXTENSIONS, IMAGE_VALENCES, IMAGES_DIR, VALENCE_LABELS_FILE
from lib.json_io import write_json_atomic
from lib.vlm_utils import call_vlm, fix_json, load_img, strip_markdown_code_format

prompt = """
Task: Visual Valence (image-only)

Analyze the image as a whole. Output one label for valence, which is just what you think the tone of the image is as a whole.

Output rules (must follow exactly):
- Output ONLY a single JSON object.
- Keys must be exactly: "valence", "confidence", "explanation".
- Values must be chosen ONLY from the allowed options below.
- you are only allowed to put free text/explanation in the "explanation" field. 

Allowed values:
- valence: "positive" | "neutral" | "negative"
- confidence: "low" | "medium" | "high"
- explanation: brief explanation of your reasoning for the valence and confidence labels.

JSON schema to follow:
{
  "valence": "<positive|neutral|negative>",
  "confidence": "<low|medium|high>", 
  "explanation": "<brief explanation of your reasoning>"
}
"""


def query_valence(img_file):
    image = load_img(IMAGES_DIR / img_file, max_size=720)
    return call_vlm(frame=image, prompt=prompt, model="gpt-5-mini", reasoning="high")


class InvalidValenceError(ValueError):
    """The reply parsed, but its valence is not one of IMAGE_VALENCES."""


def parse_label(reply: str) -> dict:
    """Parse a valence reply; raise unless it is a JSON object whose valence is in IMAGE_VALENCES."""
    label = json.loads(reply)
    if not isinstance(label, dict):
        raise ValueError(f"expected a JSON object, got {type(label).__name__}")
    if label.get("valence") not in IMAGE_VALENCES:
        raise InvalidValenceError(f"invalid valence {label.get('valence')!r}; expected one of {IMAGE_VALENCES}")
    return label


def process_image(img_file):
    return img_file, parse_label(strip_markdown_code_format(query_valence(img_file)))


def load_existing_labels() -> dict:
    """Labels already in VALENCE_LABELS_FILE; abort (leaving it untouched) if it cannot be parsed."""
    try:
        with open(VALENCE_LABELS_FILE, "r") as f:
            labels = json.load(f)
        if not isinstance(labels, dict) or not all(isinstance(v, dict) for v in labels.values()):
            raise ValueError("expected a JSON object mapping image file names to label objects")
    except (OSError, ValueError) as e:
        sys.exit(
            f"Error: cannot resume from {VALENCE_LABELS_FILE}: {e}\n"
            "It was left untouched; repair it or move it aside, or pass --overwrite to start a fresh file."
        )
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-workers", type=int, default=8, help="Images labeled in parallel")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a fresh labels file instead of keeping the labels already in it",
    )
    args = parser.parse_args()

    image_files = [
        f for f in os.listdir(IMAGES_DIR)
        if (IMAGES_DIR / f).is_file() and Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ]

    output_data = {}
    if VALENCE_LABELS_FILE.exists() and args.overwrite:
        print(f"--overwrite: starting a fresh {VALENCE_LABELS_FILE}; its current labels are not read")
    elif VALENCE_LABELS_FILE.exists():
        output_data = load_existing_labels()
    to_label = [f for f in image_files if f not in output_data]
    without_image = sorted(set(output_data) - set(image_files))
    print(
        f"Images: {len(image_files)}; labels kept from {VALENCE_LABELS_FILE.name}: {len(output_data)} "
        f"({len(without_image)} of them for files not in {IMAGES_DIR}); to label: {len(to_label)}"
    )
    failed = []
    invalid_valence = []
    lock = Lock()

    def save(img_file, result):
        with lock:
            output_data[img_file] = result
            write_json_atomic(VALENCE_LABELS_FILE, output_data, indent=2)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_image = {executor.submit(process_image, img_file): img_file for img_file in to_label}

        for future in tqdm.tqdm(as_completed(future_to_image), total=len(to_label), desc="Processing Images"):
            img_file = future_to_image[future]
            try:
                save(*future.result())
            except Exception as e:
                print(f"\nError processing {img_file}: {e}")
                # Query the model once more for this image and repair that reply's JSON
                try:
                    save(img_file, parse_label(fix_json(query_valence(img_file))))
                    print(f"Successfully fixed and saved {img_file}")
                except Exception as fix_error:
                    print(f"\nFailed to fix {img_file}: {fix_error}")
                    failed.append(img_file)
                    if isinstance(fix_error, InvalidValenceError):
                        invalid_valence.append(img_file)

    labeled = sum(1 for f in image_files if f in output_data)
    print(
        f"\nLabeled {len(to_label) - len(failed)} images this run; {labeled} of {len(image_files)} images have a "
        f"label; {len(failed)} failed: {sorted(failed)}; of these, {len(invalid_valence)} had an invalid "
        f"valence label: {sorted(invalid_valence)}"
    )
    if failed:
        print(
            f"Failed images are absent from {VALENCE_LABELS_FILE}; rerun to retry them (labeled images are kept); "
            "generate_data.py reports unlabeled images as errors."
        )


if __name__ == "__main__":
    main()
