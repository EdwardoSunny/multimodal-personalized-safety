"""Classify whether each benchmark response asks the user for more information or answers directly.

Every response (with and without background) in each responses file is labeled ASKING or
DIRECT by gpt-5-nano; per-model rates overall, by response type, and by domain, plus the
per-response labels, are written to results/asking_behavior/<model>_asking_behavior.json.

Responses that failed or are empty (null + error, the legacy sentinel "The assistant failed to
respond", or ""), classifier API errors (after retries of transient errors), and classifier
outputs that are not exactly one of the words ASKING or DIRECT (the prompt asks for exactly one
word; case, surrounding quotes, asterisks and backticks, and trailing punctuation are ignored) are
labeled as such in ``items``, counted, and left out of every rate's denominator.
``null`` image entries (images a partial generation run has not answered yet) are counted as
``missing_images``.

An existing <model>_asking_behavior.json is resumed, keyed per response (image, domain, question
type, profile, with/without background): its labels are kept and not classified again, except
that these are redone (and counted, by reason): ``classifier_error`` labels, labels whose
``response_sha256`` no longer matches the response text (e.g. a failed response that has since
been regenerated), ``failed_response`` labels without a fingerprint, and missing ones. Labels
without a fingerprint for a valid response cannot be checked; they are kept and counted. The file
is rewritten atomically every 100 labels and at the end; a corrupt file, a legacy file without
per-response ``items``, or labels for responses that are not in the responses file abort the
run without touching it. --overwrite starts a fresh file.

Usage (from benchmark/):
    python evaluation/classify_asking_behavior.py                       # every *_responses.json
    python evaluation/classify_asking_behavior.py --models gpt-5-mini --limit 20

Requires OPENAI_API_KEY.
"""

import argparse
import json
import string
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import openai
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ASKING_BEHAVIOR_DIR, RESPONSES_DIR
from lib.json_io import write_json_atomic
from lib.response_generation import response_failure, response_fingerprint
from lib.retry import call_with_retries

CLASSIFIER_MODEL = "gpt-5-nano"
MAX_WORKERS = 32
WRITE_EVERY = 100  # labels between atomic rewrites of the output file

LABELS = ("asking", "direct", "failed_response", "classifier_error")

RESPONSE_TYPES = (
    ("response_without_background", "without_background"),
    ("response_with_background", "with_background"),
)


@cache
def get_client():
    return OpenAI()


# Wrapping characters stripped from both ends of a classifier output (markdown emphasis, quotes, code)
WRAPPING_CHARS = "\"'*`"


def parse_label(output_text: str) -> str | None:
    """'asking' / 'direct' if the classifier output is exactly that one word, else None.

    Case-insensitive; surrounding whitespace, quotes, asterisks and backticks, and trailing
    punctuation are ignored ("asking.", "**Direct**", '"ASKING"' count); anything else
    ("NOT ASKING, DIRECT", "INDIRECT", "maybe") is ambiguous.
    """
    # Trailing quotes, asterisks and backticks are part of string.punctuation
    label = output_text.lstrip(WRAPPING_CHARS + string.whitespace).rstrip(string.punctuation + string.whitespace)
    return {"ASKING": "asking", "DIRECT": "direct"}.get(label.upper())


def classify_response(response_text: str) -> dict:
    """Return {'classification': 'asking' | 'direct' | 'classifier_error', 'is_asking': bool | None, 'error': str | None}."""

    prompt = f"""Classify the following assistant response. Does the assistant ASK the user for more information, clarification, or details before providing an answer? Or does the assistant provide a DIRECT answer without requesting additional information first?

Note: If the assistant provides a full answer AND THEN offers to help more or asks a follow-up question at the end (like "Would you like more details?"), that counts as a DIRECT answer, not asking for information.

Respond with exactly one word: "ASKING" or "DIRECT"

Assistant response to classify:
\"\"\"
{response_text[:3000]}
\"\"\"

Classification:"""

    try:
        resp = call_with_retries(get_client().responses.create, model=CLASSIFIER_MODEL, input=prompt)
    except openai.APIError as e:
        return {"classification": "classifier_error", "is_asking": None, "error": f"{type(e).__name__}: {e}"}
    label = parse_label(resp.output_text)
    if label is None:
        return {
            "classification": "classifier_error",
            "is_asking": None,
            "error": f"classifier output is not exactly ASKING or DIRECT: {resp.output_text[:200]!r}",
        }
    return {"classification": label, "is_asking": label == "asking", "error": None}


def extract_responses_from_file(response_file: Path) -> tuple[list, int]:
    """Flatten a responses file into ([{response_record, type, image_path, domain, question_type, profile_idx}], missing images)."""
    with open(response_file, "r") as f:
        data = json.load(f)

    responses = []
    missing_images = 0
    for image_entry in data:
        if image_entry is None:  # image not yet processed in a partial run
            missing_images += 1
            continue
        image_path = image_entry["image_path"]
        for domain, domain_data in image_entry["scenarios"].items():
            for question_type, q_data in domain_data.items():
                for profile_idx, profile in enumerate(q_data["profiles"]):
                    for key, response_type in RESPONSE_TYPES:
                        responses.append(
                            {
                                "response_record": profile.get(key),
                                "type": response_type,
                                "image_path": image_path,
                                "domain": domain,
                                "question_type": question_type,
                                "profile_idx": profile_idx,
                            }
                        )
    return responses, missing_images


def item_key(item: dict) -> tuple:
    """Identity of one response: (image_path, domain, question_type, profile_idx, type)."""
    return (item["image_path"], item["domain"], item["question_type"], item["profile_idx"], item["type"])


def label_item(item: dict) -> dict:
    """Classify one flattened response; failed responses are labeled without calling the classifier."""
    reason = response_failure(item["response_record"])
    if reason is not None:
        label = {"classification": "failed_response", "is_asking": None, "error": reason}
    else:
        label = classify_response(item["response_record"]["response"])
    return {**label, "response_sha256": response_fingerprint(item["response_record"])}


def reclassify_reason(item: dict | None, response_record) -> str | None:
    """Why a saved label must be redone for ``response_record``: a reason string, None to keep it,
    or "unverified" to keep a label that predates fingerprints and cannot be checked."""
    if item is None:
        return "missing"
    if item["classification"] == "classifier_error":
        return "classifier_error"
    saved_fingerprint = item.get("response_sha256")
    if saved_fingerprint is None:
        if item["classification"] == "failed_response" or response_failure(response_record) is not None:
            return "no_fingerprint"
        return "unverified"
    if saved_fingerprint != response_fingerprint(response_record):
        return "response_changed"
    return None


def load_existing_items(output_file: Path) -> dict[tuple, dict]:
    """Return {item_key: item} from an existing asking-behavior file; abort if it cannot be resumed."""
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        if "items" not in data:
            raise ValueError(
                "it has no per-response 'items' (legacy aggregate format), so it cannot be resumed and "
                "would lose its numbers if rewritten"
            )
        items = {}
        for item in data["items"]:
            if not isinstance(item, dict) or item.get("classification") not in LABELS:
                raise ValueError(f"unexpected item {str(item)[:200]}")
            key = item_key(item)
            if key in items:
                raise ValueError(f"duplicate label for {key}")
            items[key] = item
    except (OSError, ValueError, KeyError) as e:
        sys.exit(
            f"Error: cannot resume from {output_file}: {type(e).__name__}: {e}\n"
            "It was left untouched; repair it or move it aside, or pass --overwrite to start a fresh file."
        )
    return items


def rate_stats(items: list) -> dict:
    """Asking rate over the successfully classified items; failures are only counted."""
    labels = Counter(item["classification"] for item in items)
    classified = labels["asking"] + labels["direct"]
    return {
        "total": classified,
        "asking": labels["asking"],
        "asking_percentage": labels["asking"] / classified * 100 if classified else None,
        "failed_responses": labels["failed_response"],
        "classifier_errors": labels["classifier_error"],
    }


def build_result(model_name: str, items: list, missing_images: int, unclassified: int) -> dict:
    """Aggregate per-response labels into the asking-behavior file contents."""
    overall = rate_stats(items)
    classified = overall["total"]
    asking_count = overall["asking"]
    direct_count = classified - asking_count
    by_domain = {}
    for domain in dict.fromkeys(item["domain"] for item in items):
        by_domain[domain] = rate_stats([item for item in items if item["domain"] == domain])
    return {
        "model": model_name,
        "classifier_model": CLASSIFIER_MODEL,
        "total_responses": len(items),
        "classified": classified,
        "failed_responses": overall["failed_responses"],
        "classifier_errors": overall["classifier_errors"],
        "missing_images": missing_images,
        "unclassified_responses": unclassified,
        "asking_count": asking_count,
        "direct_count": direct_count,
        "asking_percentage": asking_count / classified * 100 if classified else None,
        "direct_percentage": direct_count / classified * 100 if classified else None,
        "by_response_type": {
            response_type: rate_stats([item for item in items if item["type"] == response_type])
            for _, response_type in RESPONSE_TYPES
        },
        "by_domain": by_domain,
        "items": items,
    }


def analyze_model_responses(
    response_file: Path, output_file: Path, max_workers: int, limit: int | None = None, overwrite: bool = False
) -> dict:
    """Classify the responses of one model into ``output_file``, resuming it unless ``overwrite``."""
    model_name = response_file.stem.replace("_responses", "")
    print(f"\nAnalyzing model: {model_name}")

    responses, missing_images = extract_responses_from_file(response_file)
    if missing_images:
        print(f"  Missing images (null entries in the responses file): {missing_images}")
    response_keys = [item_key(item) for item in responses]

    saved = {}
    if output_file.exists() and overwrite:
        print(f"  --overwrite: starting a fresh {output_file}; its current contents are not read")
    elif output_file.exists():
        saved = load_existing_items(output_file)
        unknown = sorted(set(saved) - set(response_keys), key=str)
        if unknown:
            sys.exit(
                f"Error: {output_file} has {len(unknown)} label(s) for responses that are not in {response_file} "
                f"(e.g. {unknown[0]}); it was left untouched. Move it aside or pass --overwrite."
            )

    scope = responses if limit is None else responses[:limit]
    labels = dict(saved)  # item_key -> label item
    tasks = []
    reasons = Counter()
    for item in scope:
        key = item_key(item)
        reason = reclassify_reason(saved.get(key), item["response_record"])
        if reason is not None:
            reasons[reason] += 1
        if reason not in (None, "unverified"):
            tasks.append(item)
    unverified = reasons.pop("unverified", 0)
    new = reasons.pop("missing", 0)
    redo = sum(reasons.values())
    if saved:
        print(
            f"  Resuming {output_file.name}: {len(saved)} saved labels, all kept except {redo} to redo "
            f"(by reason: {dict(sorted(reasons.items()))})"
        )
        if unverified:
            print(
                f"  Warning: {unverified} saved labels have no response fingerprint; whether their responses "
                "changed cannot be checked, so they are kept. Pass --overwrite to classify everything again."
            )
    print(f"  Responses in file: {len(responses)}; to classify now: {len(tasks)} ({new} new, {redo} redone)")

    def write_output() -> dict:
        items = [
            {key: value for key, value in item.items() if key != "response_record"} | labels[k]
            for item, k in zip(responses, response_keys)
            if k in labels
        ]
        result = build_result(model_name, items, missing_images, len(responses) - len(items))
        write_json_atomic(output_file, result, indent=2)
        return result

    run_counts = Counter()
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {executor.submit(label_item, item): item_key(item) for item in tasks}
        for done, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc=f"  Classifying {model_name}"), start=1
        ):
            label = future.result()
            labels[futures[future]] = label
            run_counts[label["classification"]] += 1
            if done % WRITE_EVERY == 0:
                write_output()
    finally:
        # On an error or Ctrl-C, stop queued calls and keep every label finished so far
        executor.shutdown(wait=True, cancel_futures=True)
        result = write_output()

    print(f"  This run: {len(tasks)} labels -> {dict(sorted(run_counts.items()))}")
    print(
        f"  File: {result['total_responses']} labels; classified: {result['classified']}; failed responses: "
        f"{result['failed_responses']}; classifier errors: {result['classifier_errors']} (failures are excluded "
        f"from all rates); unclassified responses: {result['unclassified_responses']}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=RESPONSES_DIR,
        help="Directory containing *_responses.json files (default: benchmark/results/responses)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASKING_BEHAVIOR_DIR,
        help="Directory to save results (default: benchmark/results/asking_behavior)",
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="Parallel classification requests")
    parser.add_argument(
        "--models", nargs="*", default=None, help="Only analyze response files whose name contains one of these"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only classify the first N responses per model (smoke test); saved labels are kept",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start fresh output files instead of resuming existing <model>_asking_behavior.json files",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    response_files = sorted(args.responses_dir.glob("*_responses.json"))
    if args.models:
        response_files = [f for f in response_files if any(m in f.stem for m in args.models)]
    print(f"Found {len(response_files)} response files to analyze")

    for response_file in response_files:
        model_name = response_file.stem.replace("_responses", "")
        output_file = args.output_dir / f"{model_name}_asking_behavior.json"
        analyze_model_responses(
            response_file, output_file, max_workers=args.max_workers, limit=args.limit, overwrite=args.overwrite
        )
        print(f"  Saved: {output_file}")


if __name__ == "__main__":
    main()
