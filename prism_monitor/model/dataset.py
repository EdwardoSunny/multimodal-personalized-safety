"""Split-file and cached-embedding datasets for PRISM Monitor."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "val", "test")
LABEL_KEYS = ("safety_label", "image_tone", "text_tone", "conflict_type")


class CachedEmbeddingDataset(Dataset):
    """Pre-computed encoder embeddings (written by `cache_embeddings.py`)."""

    def __init__(self, cache_path: str | Path):
        cache = torch.load(cache_path, weights_only=True)
        self.image_emb = cache["image_emb"]          # [N, D]
        self.text_emb = cache["text_emb"]            # [N, D]
        self.safety_label = cache["safety_label"]    # [N]
        self.image_tone = cache["image_tone"]        # [N]
        self.text_tone = cache["text_tone"]          # [N]
        self.conflict_type = cache["conflict_type"]  # [N]

        print(f"Loaded {len(self)} cached embeddings from {cache_path}")
        n_safe = (self.safety_label == 0).sum().item()
        print(f"  Label distribution: Safe={n_safe}, Unsafe={len(self) - n_safe}")

    def __len__(self) -> int:
        return len(self.safety_label)

    def __getitem__(self, idx: int) -> dict:
        return {
            "image_emb": self.image_emb[idx],
            "text_emb": self.text_emb[idx],
            "safety_label": self.safety_label[idx],
            "image_tone": self.image_tone[idx],
            "text_tone": self.text_tone[idx],
            "conflict_type": self.conflict_type[idx],
        }


def create_cached_data_loaders(
    cache_dir: str | Path, batch_size: int = 32, num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders from cached embeddings."""
    cache_dir = Path(cache_dir)
    loaders = []
    for split in SPLITS:
        path = cache_dir / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing cache file: {path}. Run cache_embeddings.py first.")
        ds = CachedEmbeddingDataset(path)
        loaders.append(DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
        ))
    return tuple(loaders)


def get_split_samples(data_dir: str | Path, split: str) -> list[dict]:
    """Load the samples of one split file (`<data_dir>/<split>.json`).

    Relative `image_path` entries are resolved against `data_dir`.
    """
    split_path = Path(data_dir) / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}.")
    with open(split_path) as f:
        samples = json.load(f)["data"]
    for s in samples:
        s["image_path"] = str(split_path.parent / s["image_path"])
    return samples
