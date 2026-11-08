"""Shared driver for the generate_responses_*.py scripts.

Each benchmark profile is answered twice by the model under test: once with only the
question (``response_without_background``) and once with the user's background profile
prepended (``response_with_background``). Backends differ only in how a single
(prompt, image) request is sent, which they pass in as ``call_fn``.

``call_fn`` returns the response text or raises. Transient API errors are retried with
backoff (lib/retry.py); a request that still fails, or returns no text, is kept in the
output as ``{"prompt": ..., "response": null, "error": "<reason>"}`` and counted.

An existing ``<model>_responses.json`` is resumed, never overwritten: every record in it is
kept, images not in it yet are answered, and only its failed responses (see
``response_failure``) are queried again. ``--overwrite`` starts a fresh file instead.
"""

import argparse
import copy
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image
from tqdm import tqdm

from lib.config import BENCHMARK_DATA_FILE, BENCHMARK_ROOT, RESPONSES_DIR
from lib.json_io import write_json_atomic
from lib.retry import call_with_retries

MAX_IMAGE_SIDE = 4096

# Placeholder that older versions of this code saved instead of a failed response. Results files
# produced by them (including the paper's) contain it; readers must treat it, and "", as failed.
LEGACY_FAILED_RESPONSE = "The assistant failed to respond"

RESPONSE_KEYS = ("response_without_background", "response_with_background")

_MIME_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
}


class EmptyResponseError(RuntimeError):
    """The model call succeeded but returned no text."""


def response_failure(record) -> str | None:
    """Return why a response record ``{"prompt", "response"[, "error"]}`` is unusable, or None if it is fine.

    Failed: ``response`` missing or null (current format, reason in ``error``), the legacy
    failure sentinel, or an empty / whitespace-only string.
    """
    if not isinstance(record, dict) or "response" not in record:
        return "response missing"
    text = record["response"]
    if text is None:
        return record.get("error") or "response is null"
    if not isinstance(text, str):
        return f"response is not a string ({type(text).__name__})"
    if text == LEGACY_FAILED_RESPONSE:
        return f"legacy failure sentinel {LEGACY_FAILED_RESPONSE!r}"
    if not text.strip():
        return "empty response"
    return None


def response_fingerprint(response_record) -> str:
    """SHA-256 of a response record's text (JSON-encoded, so null and "null" differ).

    Judge and asking-behavior labels store it to detect a response that changed since.
    """
    text = response_record.get("response") if isinstance(response_record, dict) else None
    return hashlib.sha256(json.dumps(text).encode("utf-8")).hexdigest()


def load_image_for_request(image_path, max_side: int = MAX_IMAGE_SIDE):
    """Return (image bytes, mime type); images larger than ``max_side`` are re-encoded as JPEG."""
    with Image.open(image_path) as img:
        if max(img.size) <= max_side:
            ext = Path(image_path).suffix.lower().lstrip(".")
            return Path(image_path).read_bytes(), _MIME_TYPES.get(ext, "image/jpeg")

        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=90, optimize=True)
        return buffer.getvalue(), "image/jpeg"


def response_file_name(model: str) -> str:
    return f"{model.replace(':', '_').replace('/', '_')}_responses.json"


def add_common_args(parser: argparse.ArgumentParser, default_model: str, default_workers: int):
    parser.add_argument("--model", default=default_model, help="Model to query (default: %(default)s)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESPONSES_DIR,
        help="Directory for <model>_responses.json (default: benchmark/results/responses)",
    )
    parser.add_argument("--max-workers", type=int, default=default_workers, help="Images processed in parallel")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images (smoke test); records already in the output file are kept",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a fresh output file. Without it, an existing <model>_responses.json is resumed: every "
        "record in it is kept, missing images are answered and failed responses are retried",
    )


def query(call_fn: Callable[[str, Path], str], prompt: str, image_path: Path) -> dict:
    """Return ``{"prompt", "response"}``, or ``{"prompt", "response": None, "error"}`` if the call failed.

    An empty or whitespace-only response counts as failed.
    """
    try:
        text = call_with_retries(call_fn, prompt, image_path)
        if not text or not text.strip():
            raise EmptyResponseError("model returned an empty or whitespace-only response")
        return {"prompt": prompt, "response": text}
    except Exception as e:
        print(f"Response failed for {image_path.name}: {type(e).__name__}: {e}")
        return {"prompt": prompt, "response": None, "error": f"{type(e).__name__}: {e}"}


def build_prompts(question: str, background_profile: dict) -> tuple[str, str]:
    """The (without background, with background) prompts for one profile."""
    prompt_no_bg = f"User question: {question}\nAssistant response:"
    prompt_with_bg = (
        f"User background: {json.dumps(background_profile, indent=2)}\n"
        f"User question: {question}\n"
        f"Consider the user's background and identify what the user truly needs based on the provided query\n"
        f"Assistant response:"
    )
    return prompt_no_bg, prompt_with_bg


def background_of(profile: dict) -> dict:
    """What the model sees "with background": every ``details`` field except Risk_reasoning."""
    return {k: v for k, v in profile["details"].items() if k != "Risk_reasoning"}


def process_image(image_data: dict, call_fn: Callable[[str, Path], str]) -> dict:
    """Query the model for every (domain, question type, profile) of one benchmark image."""
    relative_image_path = image_data["image_path"]
    image_path = BENCHMARK_ROOT / relative_image_path

    curr_result = {
        "image_path": relative_image_path,
        "metadata": image_data["metadata"],
        "scenarios": {},
    }
    for scenario in image_data["generated_scenarios"]:
        question = scenario["question"]
        domain = scenario["domain"]
        q_type = scenario["question_type"]  # question valence: positive / neutral / negative

        curr_result["scenarios"].setdefault(domain, {})
        curr_result["scenarios"][domain][q_type] = {
            "question": question,
            "profiles": [],
        }
        for profile in scenario["user_profiles"]:
            background_profile = background_of(profile)
            prompt_no_bg, prompt_with_bg = build_prompts(question, background_profile)
            response_no_bg = query(call_fn, prompt_no_bg, image_path)
            response_with_bg = query(call_fn, prompt_with_bg, image_path)

            curr_result["scenarios"][domain][q_type]["profiles"].append(
                {
                    "background": background_profile,
                    "response_without_background": response_no_bg,
                    "response_with_background": response_with_bg,
                }
            )

    return curr_result


def check_existing_record(image_data: dict, record: dict) -> None:
    """Raise ValueError unless ``record`` (from an existing responses file) answers exactly ``image_data``."""
    image_path = image_data["image_path"]
    expected = sorted((s["domain"], s["question_type"]) for s in image_data["generated_scenarios"])
    found = sorted((domain, q_type) for domain, question_types in record["scenarios"].items() for q_type in question_types)
    if found != expected:
        raise ValueError(f"{image_path}: scenarios {found} in the responses file but {expected} in {BENCHMARK_DATA_FILE}")
    for scenario in image_data["generated_scenarios"]:
        saved = record["scenarios"][scenario["domain"]][scenario["question_type"]]
        where = f"{image_path} ({scenario['domain']}, {scenario['question_type']})"
        if saved["question"] != scenario["question"]:
            raise ValueError(f"{where}: the question differs from {BENCHMARK_DATA_FILE}")
        if len(saved["profiles"]) != len(scenario["user_profiles"]):
            raise ValueError(
                f"{where}: {len(saved['profiles'])} profiles in the responses file but "
                f"{len(scenario['user_profiles'])} in {BENCHMARK_DATA_FILE}"
            )
        for saved_profile, profile in zip(saved["profiles"], scenario["user_profiles"]):
            if saved_profile["background"] != background_of(profile):
                raise ValueError(f"{where}: a profile background differs from {BENCHMARK_DATA_FILE}")


def retry_failed_responses(image_data: dict, record: dict, call_fn: Callable[[str, Path], str]) -> tuple[dict, int, int]:
    """Copy of an existing image ``record`` in which each failed response is queried again.

    Every other response is kept as it is. Returns ``(record, retried, recovered)``: how many
    failed responses were queried again and how many of them now have a response.
    """
    image_path = BENCHMARK_ROOT / image_data["image_path"]
    record = copy.deepcopy(record)
    retried = recovered = 0
    for scenario in image_data["generated_scenarios"]:
        saved = record["scenarios"][scenario["domain"]][scenario["question_type"]]
        for saved_profile, profile in zip(saved["profiles"], scenario["user_profiles"]):
            prompts = build_prompts(scenario["question"], background_of(profile))
            for key, prompt in zip(RESPONSE_KEYS, prompts):
                if response_failure(saved_profile.get(key)) is None:
                    continue
                retried += 1
                saved_profile[key] = query(call_fn, prompt, image_path)
                if response_failure(saved_profile[key]) is None:
                    recovered += 1
    return record, retried, recovered


def collect_failures(results: list) -> list[dict]:
    """List every failed response in ``results`` with its location and reason."""
    failures = []
    for image_result in results:
        for domain, question_types in image_result["scenarios"].items():
            for q_type, scenario in question_types.items():
                for profile_idx, profile in enumerate(scenario["profiles"]):
                    for key in RESPONSE_KEYS:
                        reason = response_failure(profile.get(key))
                        if reason is not None:
                            failures.append(
                                {
                                    "image_path": image_result["image_path"],
                                    "domain": domain,
                                    "question_type": q_type,
                                    "profile_idx": profile_idx,
                                    "response_type": key,
                                    "error": reason,
                                }
                            )
    return failures


def load_existing_responses(output_file: Path) -> dict[str, dict]:
    """Return {image_path: record} from an existing responses file; abort if it cannot be parsed.

    ``null`` entries (images a partial run had not answered yet) are not records; those images
    are answered again.
    """
    try:
        with open(output_file, "r") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError("expected a JSON list of image records")
        by_path = {}
        for record in records:
            if record is None:
                continue
            if not (
                isinstance(record, dict)
                and isinstance(record.get("image_path"), str)
                and isinstance(record.get("scenarios"), dict)
            ):
                raise ValueError("expected image records with 'image_path' and 'scenarios'")
            if record["image_path"] in by_path:
                raise ValueError(f"duplicate record for {record['image_path']}")
            by_path[record["image_path"]] = record
    except (OSError, ValueError) as e:
        sys.exit(
            f"Error: cannot resume from {output_file}: {e}\n"
            "It was left untouched; repair it or move it aside, or pass --overwrite to start a fresh file."
        )
    return by_path


def generate_responses(
    call_fn: Callable[[str, Path], str],
    model: str,
    output_dir: Path,
    max_workers: int,
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Answer the benchmark with ``call_fn`` into ``<output_dir>/<model>_responses.json``.

    An existing output file is resumed unless ``overwrite``: every record in it is kept, images
    missing from it (among the first ``limit``) are answered, and failed responses of its
    records (among the first ``limit`` images) are queried again; nothing else in it changes.
    A file that cannot be parsed or does not match benchmark_data.json aborts the run untouched.
    Records follow the order of benchmark_data.json, with ``null`` for images of this run not
    finished yet; the file is rewritten atomically after every image. Failed responses stay in
    the file, marked with ``"response": null`` and an ``error``; their count and locations also
    go to ``<model>_responses_failures.json``.
    """
    with open(BENCHMARK_DATA_FILE, "r") as f:
        all_data = json.load(f)
    data = all_data if limit is None else all_data[:limit]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / response_file_name(model)
    failures_file = output_file.with_name(output_file.stem + "_failures.json")

    existing = {}
    if output_file.exists() and overwrite:
        print(f"--overwrite: starting a fresh {output_file}; its current contents are not read")
    elif output_file.exists():
        existing = load_existing_responses(output_file)
        benchmark_images = {image_data["image_path"] for image_data in all_data}
        unknown = sorted(set(existing) - benchmark_images)
        if unknown:
            sys.exit(
                f"Error: {output_file} has {len(unknown)} image(s) that are not in {BENCHMARK_DATA_FILE} "
                f"(e.g. {unknown[0]}); it was left untouched. Move it aside or pass --overwrite."
            )
        for image_data in all_data:
            if image_data["image_path"] in existing:
                try:
                    check_existing_record(image_data, existing[image_data["image_path"]])
                except (KeyError, TypeError, ValueError) as e:
                    sys.exit(
                        f"Error: cannot resume from {output_file}: {type(e).__name__}: {e}\n"
                        f"It does not match {BENCHMARK_DATA_FILE} and was left untouched. "
                        "Move it aside or pass --overwrite."
                    )

    # image_path -> record; None marks an image of this run that is not finished yet
    results = dict(existing)
    new_images, retry_images = [], []
    failed_in_existing = kept_unchanged = 0
    for image_data in data:
        record = existing.get(image_data["image_path"])
        if record is None:
            results[image_data["image_path"]] = None
            new_images.append(image_data)
            continue
        n_failed = len(collect_failures([record]))
        if n_failed:
            retry_images.append(image_data)
            failed_in_existing += n_failed
        else:
            kept_unchanged += 1
    outside_limit = len(existing) - kept_unchanged - len(retry_images)

    if existing:
        print(f"Resuming {output_file}: all {len(existing)} image records in it are kept")
        print(f"  unchanged: {kept_unchanged} complete + {outside_limit} outside --limit")
        print(f"  with failed responses to retry: {len(retry_images)} images ({failed_in_existing} responses)")
    print(f"New images to answer: {len(new_images)}")

    def write_output():
        ordered = [results[d["image_path"]] for d in all_data if d["image_path"] in results]
        write_json_atomic(output_file, ordered, indent=2)

    retried = recovered = 0
    if new_images or retry_images:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = {executor.submit(process_image, d, call_fn): (d, False) for d in new_images}
            for d in retry_images:
                futures[executor.submit(retry_failed_responses, d, existing[d["image_path"]], call_fn)] = (d, True)
            with tqdm(total=len(futures), desc="Processing images") as pbar:
                for future in as_completed(futures):
                    image_data, is_retry = futures[future]
                    if is_retry:
                        record, n_retried, n_recovered = future.result()
                        retried += n_retried
                        recovered += n_recovered
                    else:
                        record = future.result()
                    results[image_data["image_path"]] = record
                    write_output()
                    pbar.update(1)
        finally:
            # On an error or Ctrl-C, stop queued images; finished ones are already on disk
            executor.shutdown(wait=True, cancel_futures=True)

    final_records = [record for record in results.values() if record is not None]
    total_responses = sum(
        len(scenario["profiles"]) * len(RESPONSE_KEYS)
        for image_result in final_records
        for question_types in image_result["scenarios"].values()
        for scenario in question_types.values()
    )
    failures = collect_failures(final_records)
    write_json_atomic(
        failures_file,
        {
            "model": model,
            "total_responses": total_responses,
            "failed_responses": len(failures),
            "failures": failures,
        },
        indent=2,
    )

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(
        f"Images in the file: {len(final_records)} = {len(existing)} resumed "
        f"({len(retry_images)} of them with retried responses) + {len(new_images)} new"
    )
    print(f"Retried failed responses: {retried}; now answered: {recovered}; still failed: {retried - recovered}")
    print(f"Total responses: {total_responses}; failed: {len(failures)} (listed in {failures_file})")
    print(f"{'=' * 80}")
    return output_file
