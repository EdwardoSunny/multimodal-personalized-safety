"""Face blurring for collected images (OpenCV YuNet detector + strong Gaussian blur / mosaic).

Detector weights are fetched once from the Hugging Face Hub (`opencv/face_detection_yunet`,
pinned revision) into the local HF cache.

The detector also fires on some non-faces (e.g. wounds, skin close-ups, animal faces), which
would blur the very hazard an image is meant to show. Review the detections and pass
images whose detections are all false positives via --exclude.

Usage as a script (blurs every image in a folder in place, or into --output-dir):
    python collection/face_blur.py data/images --output-dir data/images_blurred --exclude 615.jpg 618.jpg

Files with an image extension (lib/config.IMAGE_EXTENSIONS) that OpenCV cannot read are
neither blurred nor copied: they are listed at the end and the script exits with status 1
after processing every other image. Other files and sub-folders are listed and left alone.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

from lib.config import IMAGE_EXTENSIONS

YUNET_REPO = "opencv/face_detection_yunet"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_REVISION = "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310"

SCORE_THRESHOLD = 0.6  # below YuNet's 0.9 default: favor recall for privacy
DETECT_MAX_SIDE = 1600  # downscale very large images for detection
DETECT_MIN_SIDE = 640  # upscale small images so small faces are found
PAD = 0.15  # grow each box by this fraction on every side

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        weights = hf_hub_download(YUNET_REPO, YUNET_FILE, revision=YUNET_REVISION)
        _detector = cv2.FaceDetectorYN.create(weights, "", (320, 320), SCORE_THRESHOLD)
    return _detector


def detect_faces(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Face boxes as (x1, y1, x2, y2) in original image coordinates."""
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    scale = 1.0
    if longest > DETECT_MAX_SIDE:
        scale = DETECT_MAX_SIDE / longest
    elif longest < DETECT_MIN_SIDE:
        scale = DETECT_MIN_SIDE / longest
    resized = img_bgr if scale == 1.0 else cv2.resize(img_bgr, (round(w * scale), round(h * scale)))

    detector = _get_detector()
    detector.setInputSize((resized.shape[1], resized.shape[0]))
    _, faces = detector.detect(resized)
    if faces is None:
        return []
    boxes = []
    for x, y, bw, bh in faces[:, :4] / scale:
        boxes.append((int(x), int(y), int(x + bw), int(y + bh)))
    return boxes


def blur_boxes(img_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]], pad: float = PAD) -> np.ndarray:
    """Blur each box (padded): mosaic for tiny faces, full-size Gaussian otherwise."""
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    for x1, y1, x2, y2 in boxes:
        pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
        x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
        x2, y2 = min(w, x2 + pw), min(h, y2 + ph)
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        rh, rw = roi.shape[:2]
        if min(rh, rw) < 30:
            s = max(4, min(rh, rw) // 3)
            small = cv2.resize(roi, (max(1, rw // s), max(1, rh // s)))
            out[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        else:
            k = min(rh, rw) - 1
            k = k if k % 2 else k - 1
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0, borderType=cv2.BORDER_REPLICATE)
    return out


def blur_faces(img_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (blurred image, number of faces blurred)."""
    boxes = detect_faces(img_bgr)
    return (blur_boxes(img_bgr, boxes) if boxes else img_bgr), len(boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Folder of images")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write here instead of overwriting in place")
    parser.add_argument("--exclude", nargs="*", default=[], help="File names to leave unblurred (reviewed false positives)")
    args = parser.parse_args()
    exclude = set(args.exclude)

    out_dir = args.output_dir or args.input_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    n_images = n_blurred = n_faces = 0
    unreadable, not_images = [], []
    for path in sorted(args.input_dir.iterdir()):
        if not (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            not_images.append(path.name)
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"Unreadable image (not blurred, not copied): {path}")
            unreadable.append(path.name)
            continue
        n_images += 1
        blurred, faces = (img, 0) if path.name in exclude else blur_faces(img)
        if faces:
            n_blurred += 1
            n_faces += faces
            if not cv2.imwrite(str(out_dir / path.name), blurred):
                sys.exit(f"Failed to write {out_dir / path.name}")
        elif out_dir != args.input_dir:
            shutil.copy2(path, out_dir / path.name)
    print(f"{n_images} images, {n_blurred} with faces, {n_faces} faces blurred -> {out_dir}")
    if not_images:
        print(f"{len(not_images)} entries without an image extension left alone: {not_images}")
    if unreadable:
        sys.exit(f"Error: {len(unreadable)} unreadable image file(s) were not processed: {unreadable}")


if __name__ == "__main__":
    main()
