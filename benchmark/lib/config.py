"""Filesystem layout of the benchmark. Every path is relative to ``benchmark/``."""

from pathlib import Path

# benchmark/ directory (parent of lib/)
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent

# Dataset (downloaded with scripts/download_data.py, or rebuilt with collection/ + generation/)
DATA_DIR = BENCHMARK_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
FAILED_IMAGES_DIR = DATA_DIR / "images_failed"
BENCHMARK_DATA_FILE = DATA_DIR / "benchmark_data.json"
VALENCE_LABELS_FILE = DATA_DIR / "image_valence_labels.json"

# Image file extensions accepted at every stage (gathering, valence labeling, scenario generation)
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"})

# Allowed image valence labels (data/image_valence_labels.json, metadata.valence in benchmark_data.json)
IMAGE_VALENCES = ("positive", "neutral", "negative")

# Raw image sources used to rebuild the dataset
SOURCE_DATA_DIR = BENCHMARK_ROOT / "collection" / "source_data"

# Domain definitions used for scenario generation
DOMAIN_DEFS_JSON = BENCHMARK_ROOT / "domain_definitions.json"

# Evaluation outputs
RESULTS_DIR = BENCHMARK_ROOT / "results"
RESPONSES_DIR = RESULTS_DIR / "responses"
JUDGE_RESULTS_DIR = RESULTS_DIR / "judge_results"
ASKING_BEHAVIOR_DIR = RESULTS_DIR / "asking_behavior"
