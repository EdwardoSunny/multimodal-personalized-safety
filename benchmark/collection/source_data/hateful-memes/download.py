"""Download the Facebook Hateful Memes Kaggle dataset into ./data/ (next to this script).

Requires Kaggle credentials (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json).
"""

import argparse
from pathlib import Path

import kagglehub

OUTDIR = Path(__file__).resolve().parent / "data"

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    path = kagglehub.dataset_download("parthplc/facebook-hateful-meme-dataset", output_dir=str(OUTDIR))
    print("Path to dataset files:", path)
