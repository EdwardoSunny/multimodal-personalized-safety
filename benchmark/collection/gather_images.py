#!/usr/bin/env python3
"""Gather benchmark images from collection/source_data/, filter them, and save them as data/images/<i>.<ext>.

Sampling is spread evenly across sources (each top-level folder of source_data/, and each
subreddit under reddit/images/ as its own source), then evenly across each source's
sub-folders. Every sampled image is re-encoded, with detected faces blurred (face_blur.py)
according to --face-blur: `reddit` (default) blurs only images from the Reddit sources
(reddit/images/<subreddit>, source names "reddit_<subreddit>"), as in the released dataset;
`all` blurs every image; `none` blurs nothing. Each image is then checked by the privacy,
spam, and quality filters in filter.py (gpt-5-mini); batches are drawn until --max-samples
images pass, and the final set is again selected evenly across sources. Images that fail a
filter go to data/images_failed/. Both folders get a manifest mapping output names to source
files. Review blurred images afterwards: the face detector can fire on wounds or skin
close-ups (see face_blur.py --exclude).

Usage (from benchmark/):
    python collection/gather_images.py --max-samples 650

Requires OPENAI_API_KEY. Refuses to start if data/images/ or data/images_failed/ already
contains files (e.g. the downloaded benchmark), unless --overwrite is passed; with it,
same-named files are replaced and files from before that were not replaced are listed.
"""

import argparse
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collection.face_blur import blur_faces
from collection.filter import check_violation, privacy_prompt, quality_prompt, spam_prompt
from lib.config import FAILED_IMAGES_DIR, IMAGE_EXTENSIONS, IMAGES_DIR, SOURCE_DATA_DIR
from lib.json_io import write_text_atomic

MAX_ITERATIONS = 10

# Each reddit/images/<subreddit> folder is the source "reddit_<subreddit>"
REDDIT_SOURCE_PREFIX = "reddit_"
# --face-blur: blur faces in Reddit-source images only (as in the release), in every image, or in none
FACE_BLUR_MODES = ("reddit", "all", "none")


def find_all_images(directory):
    """Recursively find all image files in a directory."""
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(Path(root) / file)
    return images


def _add_source(source_categories, source_name, source_folder, indent):
    images = find_all_images(source_folder)
    if not images:
        print(f"{indent}No images found in {source_name}")
        return

    # Group images by their sub-folder (category) within the source
    for img in images:
        relative_parent = img.parent.relative_to(source_folder)
        category_key = str(relative_parent) if str(relative_parent) != '.' else 'root'
        source_categories[source_name][category_key].append(img)

    total_in_source = sum(len(imgs) for imgs in source_categories[source_name].values())
    print(f"{indent}Found {total_in_source} images across {len(source_categories[source_name])} categories")
    for cat, imgs in source_categories[source_name].items():
        print(f"{indent}  {cat}: {len(imgs)} images")


def organize_images_by_source(source_dir):
    """
    Returns a nested dict: {source_name: {category_path: [image_paths]}}

    Each reddit/images/<subreddit> folder is treated as its own source.
    """
    source_categories = defaultdict(lambda: defaultdict(list))

    for source_folder in sorted(source_dir.iterdir()):
        if not source_folder.is_dir():
            continue

        if source_folder.name == "reddit":
            reddit_images_dir = source_folder / "images"
            if reddit_images_dir.is_dir():
                print("Scanning reddit subreddits...")
                for subreddit_folder in sorted(reddit_images_dir.iterdir()):
                    if subreddit_folder.is_dir():
                        subreddit_name = f"{REDDIT_SOURCE_PREFIX}{subreddit_folder.name}"
                        print(f"  Scanning {subreddit_name}...")
                        _add_source(source_categories, subreddit_name, subreddit_folder, indent="    ")
            continue

        if source_folder.name.startswith(REDDIT_SOURCE_PREFIX):
            # Such a name would be indistinguishable from a subreddit source (and blurred as one)
            raise ValueError(f"Rename {source_folder}: names starting with {REDDIT_SOURCE_PREFIX!r} are reserved for subreddits")
        print(f"Scanning {source_folder.name}...")
        _add_source(source_categories, source_folder.name, source_folder, indent="  ")

    return source_categories


def distribute_samples(source_categories, max_samples):
    """
    Distribute max_samples evenly across sources first, then evenly across each
    source's categories, picking evenly spaced images within a category.
    Returns a dict: {source_name: {category: [selected_images]}}
    """
    if not source_categories:
        print("No images found!")
        return {}

    num_sources = len(source_categories)
    base_per_source = max_samples // num_sources
    extra_samples = max_samples % num_sources

    print("\n=== Distribution Plan ===")
    print(f"Total sources: {num_sources}")
    print(f"Base samples per source: {base_per_source}")
    print(f"Extra samples to distribute: {extra_samples}")
    print()

    source_allocations = {}
    for i, source_name in enumerate(source_categories):
        source_allocations[source_name] = base_per_source + (1 if i < extra_samples else 0)
        print(f"{source_name}: allocated {source_allocations[source_name]} samples")
    print()

    selected_images = defaultdict(lambda: defaultdict(list))
    total_selected = 0

    for source_name, categories in source_categories.items():
        source_allocation = source_allocations[source_name]
        num_categories = len(categories)
        if num_categories == 0:
            continue

        base_per_category = source_allocation // num_categories
        extra_in_source = source_allocation % num_categories

        print(f"{source_name} ({source_allocation} samples across {num_categories} categories):")

        for i, (category, images) in enumerate(categories.items()):
            n_samples = min(base_per_category + (1 if i < extra_in_source else 0), len(images))
            if n_samples > 0:
                step = len(images) / n_samples
                selected = [images[int(j * step)] for j in range(n_samples)]
                selected_images[source_name][category] = selected
                total_selected += len(selected)
                print(f"  {category}: {len(selected)} images (from {len(images)} available)")
            else:
                print(f"  {category}: 0 images (requested {n_samples})")
        print()

    print("=== Final Distribution ===")
    for source_name, categories in selected_images.items():
        print(f"{source_name}: {sum(len(imgs) for imgs in categories.values())} images")
        for category, images in categories.items():
            print(f"  {category}: {len(images)} images")
    print(f"\nTotal selected: {total_selected} images")

    return selected_images


def process_single_image(args):
    """
    Re-encode one image into the temp directory (blurring faces if requested) and run the
    privacy, spam, and quality filters.
    Returns (success, passed_filters, idx, error_msg)
    """
    idx, img_path, dest_path, face_blur = args

    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return (False, False, idx, f"Could not read {img_path}")
        if face_blur:
            img_bgr, _ = blur_faces(img_bgr)
        cv2.imwrite(str(dest_path), img_bgr)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(check_violation, prompt, str(dest_path))
                for prompt in (privacy_prompt, spam_prompt, quality_prompt)
            ]
            gates = [f.result() for f in futures]  # each (keep, reason, error)

        errors = [error for _, _, error in gates if error]
        if errors:
            return (False, False, idx, "; ".join(errors))

        # Image passes only if all three filters say to keep it
        return (True, all(keep for keep, _, _ in gates), idx, None)

    except Exception as e:
        # Return a string, not the exception, to avoid pickling issues with multiprocessing
        return (False, False, idx, f"{type(e).__name__}: {str(e)}")


def should_blur(source_name, face_blur_mode):
    """Whether images of ``source_name`` get face blurring under ``face_blur_mode`` (see FACE_BLUR_MODES)."""
    if face_blur_mode not in FACE_BLUR_MODES:
        raise ValueError(f"face_blur_mode must be one of {FACE_BLUR_MODES}, got {face_blur_mode!r}")
    return face_blur_mode == "all" or (face_blur_mode == "reddit" and source_name.startswith(REDDIT_SOURCE_PREFIX))


def filter_images(selected_images, temp_dir, face_blur_mode="reddit"):
    """
    Copy selected images into temp_dir and filter them in parallel.
    Returns (passed images, failed images, all attempted source images); each passed/failed
    image is (img_path, source_name, category, temp_file_path).
    """
    all_images = [
        (img, source_name, category)
        for source_name, categories in selected_images.items()
        for category, images in categories.items()
        for img in images
    ]

    temp_dir.mkdir(parents=True, exist_ok=True)
    # Unique temp filenames so batches from different iterations never collide
    timestamp = int(time.time() * 1000000)
    tasks = [
        (
            idx,
            img_path,
            temp_dir / f"temp_{timestamp}_{idx}{img_path.suffix}",
            should_blur(source_name, face_blur_mode),
        )
        for idx, (img_path, source_name, _) in enumerate(all_images)
    ]
    n_blur = sum(task[3] for task in tasks)

    # The filters are I/O-bound API calls, so use many workers
    num_workers = min(30, max(1, cpu_count() * 2))
    print(f"\n=== Processing and filtering {len(all_images)} images ===")
    print(f"Using {num_workers} parallel workers")
    print(
        f"Face blurring ({face_blur_mode}): {n_blur} of {len(tasks)} images; "
        "applying privacy, spam, and quality filters...\n"
    )

    success_count = 0
    error_count = 0
    errors = []
    passed_images = []
    failed_images = []
    used_source_images = []

    with Pool(num_workers) as pool:
        for success, passed_filters, idx, error_msg in tqdm(
            pool.imap_unordered(process_single_image, tasks),
            total=len(all_images),
            desc="Processing images",
            unit="img",
        ):
            img_path, source_name, category = all_images[idx]
            # Track every attempted image (including broken ones) so it is never retried
            used_source_images.append(img_path)

            if success:
                success_count += 1
                entry = (img_path, source_name, category, tasks[idx][2])
                (passed_images if passed_filters else failed_images).append(entry)
            else:
                error_count += 1
                if len(errors) < 10:
                    errors.append(error_msg)

    print("\n=== Processing Summary ===")
    print(f"Successfully processed: {success_count} images")
    print(f"Passed filters: {len(passed_images)} images")
    print(f"Failed filters: {len(failed_images)} images")
    if error_count > 0:
        print(f"Processing errors: {error_count} images")
        print(f"First {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")

    return passed_images, failed_images, used_source_images


def select_images_evenly(all_passed_images, max_samples):
    """
    Select max_samples images evenly distributed across sources.
    Returns (selected_images, excess_images)
    """
    if len(all_passed_images) <= max_samples:
        return all_passed_images, []

    print(f"\n=== Selecting {max_samples} images evenly from {len(all_passed_images)} passed images ===")

    source_groups = defaultdict(list)
    for img_data in all_passed_images:
        source_groups[img_data[1]].append(img_data)

    num_sources = len(source_groups)
    base_per_source = max_samples // num_sources
    extra_samples = max_samples % num_sources

    print(f"Distributing across {num_sources} sources:")

    selected = []
    for i, (source_name, source_images) in enumerate(sorted(source_groups.items())):
        n_from_source = min(base_per_source + (1 if i < extra_samples else 0), len(source_images))
        if n_from_source > 0:
            step = len(source_images) / n_from_source
            selected.extend(source_images[int(j * step)] for j in range(n_from_source))
            print(f"  {source_name}: {n_from_source} images (from {len(source_images)} available)")

    selected_set = set(selected)
    excess = [img for img in all_passed_images if img not in selected_set]

    print(f"Selected: {len(selected)} images, Excess: {len(excess)} images")
    return selected, excess


def save_final_images(selected_images, failed_images, output_dir, failed_dir, max_samples):
    """
    Move selected images to output_dir as <i><ext> and failed images to failed_dir as
    failed_<i><ext>, writing a manifest for each.
    """
    print("\n=== Saving final images ===")
    output_dir.mkdir(exist_ok=True, parents=True)
    failed_dir.mkdir(exist_ok=True, parents=True)

    manifest_entries = []
    for idx, (img_path, source_name, category, temp_path) in enumerate(selected_images):
        if not temp_path.exists():
            raise FileNotFoundError(f"Processed copy of selected image {img_path} is missing: {temp_path}")
        shutil.move(str(temp_path), str(output_dir / f"{idx}{img_path.suffix}"))
        manifest_entries.append((idx, img_path, source_name, category))

    manifest_path = output_dir / "manifest.txt"
    manifest_lines = [f"Target samples: {max_samples}\n", f"Actual samples: {len(manifest_entries)}\n\n"]
    manifest_lines += [
        f"{idx}{img_path.suffix} <- {source_name}/{category}/{img_path.name}\n"
        for idx, img_path, source_name, category in manifest_entries
    ]
    write_text_atomic(manifest_path, "".join(manifest_lines))

    print(f"Saved {len(manifest_entries)} images to {output_dir}")
    print(f"Manifest written to {manifest_path}")

    failed_manifest_entries = []
    for idx, (img_path, source_name, category, temp_path) in enumerate(failed_images):
        if not temp_path.exists():
            raise FileNotFoundError(f"Processed copy of rejected image {img_path} is missing: {temp_path}")
        shutil.move(str(temp_path), str(failed_dir / f"failed_{idx}{img_path.suffix}"))
        failed_manifest_entries.append((idx, img_path, source_name, category))

    if failed_manifest_entries:
        failed_manifest_path = failed_dir / "manifest_failed.txt"
        write_text_atomic(
            failed_manifest_path,
            "".join(
                f"failed_{idx}{img_path.suffix} <- {source_name}/{category}/{img_path.name}\n"
                for idx, img_path, source_name, category in failed_manifest_entries
            ),
        )
        print(f"Saved {len(failed_manifest_entries)} failed images to {failed_dir}")
        print(f"Failed manifest written to {failed_manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-samples", type=int, default=650, help="Number of images to keep (default: %(default)s)")
    parser.add_argument(
        "--face-blur",
        choices=FACE_BLUR_MODES,
        default="reddit",
        help="Blur detected faces in images from Reddit sources only (default, as in the released dataset), "
        "in all images, or in none",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into non-empty data/images/ and data/images_failed/ (same-named files are replaced)",
    )
    args = parser.parse_args()
    max_samples = args.max_samples

    # The output names (0.jpg, 1.png, ..., manifest.txt) are those of the released benchmark images
    existing = sorted(p for d in (IMAGES_DIR, FAILED_IMAGES_DIR) if d.is_dir() for p in d.iterdir() if p.is_file())
    if existing and not args.overwrite:
        sys.exit(
            f"Error: {len(existing)} files already in {IMAGES_DIR} and/or {FAILED_IMAGES_DIR} (e.g. {existing[0].name}); "
            "this run would replace them. Move them aside, or pass --overwrite."
        )

    print("=== Image Gathering Script with Filtering ===")
    print(f"Source directory: {SOURCE_DATA_DIR}")
    print(f"Output directory (passed): {IMAGES_DIR}")
    print(f"Output directory (failed): {FAILED_IMAGES_DIR}")
    print(f"Target samples: {max_samples}")
    print()

    source_categories = organize_images_by_source(SOURCE_DATA_DIR)
    if not source_categories:
        print(f"No images found in {SOURCE_DATA_DIR}!")
        return

    # Draw and filter batches until enough images pass (first batch is 2x the target,
    # later batches 1.5x the remaining shortfall)
    all_passed_images = []
    all_failed_images = []
    all_used_images = set()
    temp_dir = IMAGES_DIR / "temp"
    iteration = 1

    while len(all_passed_images) < max_samples:
        remaining_needed = max_samples - len(all_passed_images)
        batch_size = int(max_samples * 2.0) if iteration == 1 else int(remaining_needed * 1.5)

        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration}: Gathering batch of {batch_size} images")
        print(f"Current passed: {len(all_passed_images)}/{max_samples}")
        print(f"{'=' * 60}\n")

        available_categories = defaultdict(lambda: defaultdict(list))
        for source_name, categories in source_categories.items():
            for category, images in categories.items():
                available_images = [img for img in images if img not in all_used_images]
                if available_images:
                    available_categories[source_name][category] = available_images

        if not available_categories:
            print("No more images available to process!")
            break

        selected_images = distribute_samples(available_categories, batch_size)
        if not selected_images:
            print("No images selected!")
            break

        passed_batch, failed_batch, used_images = filter_images(selected_images, temp_dir, args.face_blur)
        all_passed_images.extend(passed_batch)
        all_failed_images.extend(failed_batch)
        all_used_images.update(used_images)

        print(f"\nIteration {iteration} complete: {len(passed_batch)} images passed filters")
        print(f"Total passed so far: {len(all_passed_images)}")

        if len(all_passed_images) >= max_samples:
            print(f"\nEnough images gathered! {len(all_passed_images)} total passed filters.")
            break

        iteration += 1
        if iteration > MAX_ITERATIONS:
            print(f"\nReached maximum iterations. Stopping with {len(all_passed_images)} passed images.")
            break

    print(f"\n{'=' * 60}")
    print(f"Processing complete: {len(all_passed_images)} total images passed filters")
    print(f"{'=' * 60}")

    if not all_passed_images:
        print("No images passed filters!")
        return

    selected_images, excess_images = select_images_evenly(all_passed_images, max_samples)
    if excess_images:
        print(f"\nDiscarding {len(excess_images)} excess images (passed filters but not selected)")

    save_final_images(selected_images, all_failed_images, IMAGES_DIR, FAILED_IMAGES_DIR, max_samples)
    written = {IMAGES_DIR / f"{idx}{entry[0].suffix}" for idx, entry in enumerate(selected_images)}
    written |= {FAILED_IMAGES_DIR / f"failed_{idx}{entry[0].suffix}" for idx, entry in enumerate(all_failed_images)}
    written.add(IMAGES_DIR / "manifest.txt")
    if all_failed_images:
        written.add(FAILED_IMAGES_DIR / "manifest_failed.txt")
    leftover = [p for p in existing if p not in written]
    replaced = len(existing) - len(leftover)

    # Removes the temp directory, including excess images
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"Final result: {len(selected_images)} images evenly distributed across sources")
    print(f"Failed directory contains {len(all_failed_images)} images that failed filters")
    print(f"Output saved to {IMAGES_DIR}")
    if existing:
        print(f"--overwrite: {replaced} files that existed before were replaced")
        if leftover:
            print(
                f"Warning: {len(leftover)} files from before were not replaced and are not in this run's "
                f"manifests; review or move them: {[str(p) for p in leftover]}"
            )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
