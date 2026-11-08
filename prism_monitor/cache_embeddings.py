#!/usr/bin/env python3
"""Pre-compute frozen encoder embeddings for the train/val/test splits.

PRISM is trained and evaluated on these cached embeddings. Writes
`cache/<encoder>/{train,val,test}.pt` (`--cache_dir` sets another directory); for Qwen3-VL also
`cache/qwen3vl/encoder_projection.safetensors` (the frozen random projection
that `train_models.py` copies next to each trained checkpoint).

Usage:
    uv run python cache_embeddings.py                          # all four encoders
    uv run python cache_embeddings.py --encoders siglip_so400m # just SigLIP
    uv run python cache_embeddings.py --batch_size 16          # smaller batches

Refuses to run if any selected encoder already has cache files (a new Qwen3-VL cache draws a
new random projection, which would no longer match checkpoints trained on the old one), unless
--overwrite is passed. Unreadable images raise an error.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from atomic_io import atomic_output
from config import get_config
from model.dataset import SPLITS, get_split_samples
from model.encoders import ENCODER_REGISTRY, PROJECTION_FILE, create_encoder, load_image

# Per-encoder batch size (sized for a large-memory GPU); override with --batch_size if OOM.
DEFAULT_BATCH_SIZES = {"siglip_so400m": 256, "clip": 256, "evaclip": 128, "qwen3vl": 256}


class SplitImageDataset(Dataset):
    """PIL images + raw text + labels from a split, for parallel CPU loading."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            image = load_image(s["image_path"])
        except Exception as e:
            raise RuntimeError(f"Cannot read image {s['image_path']}: {type(e).__name__}: {e}") from e
        return {
            "image": image,
            "text": s["question"],
            "safety_label": s["needs_background_label"],
            "image_tone": s["image_tone"] + 1,
            "text_tone": s["text_tone"] + 1,
            "conflict_type": s["conflict_type"],
        }


def collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "texts": [b["text"] for b in batch],
        "safety_label": torch.tensor([b["safety_label"] for b in batch], dtype=torch.float32),
        "image_tone": torch.tensor([b["image_tone"] for b in batch], dtype=torch.long),
        "text_tone": torch.tensor([b["text_tone"] for b in batch], dtype=torch.long),
        "conflict_type": torch.tensor([b["conflict_type"] for b in batch], dtype=torch.long),
    }


def cache_encoder(encoder_name: str, data_dir: Path, cache_dir: Path, device: torch.device,
                  batch_size: int, num_workers: int):
    """Extract and save embeddings for one encoder across all splits."""
    print(f"\n{'=' * 60}")
    print(f"  Caching: {ENCODER_REGISTRY[encoder_name]['display_name']} ({encoder_name})")
    print(f"  Batch size: {batch_size}, Workers: {num_workers}")
    print(f"{'=' * 60}")

    encoder = create_encoder(encoder_name).to(device)
    out_dir = cache_dir / encoder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if encoder.projection is not None:
        state = {k: v.detach().cpu().contiguous() for k, v in encoder.projection.state_dict().items()}
        with atomic_output(out_dir / PROJECTION_FILE) as tmp_path:
            save_file(state, str(tmp_path))

    for split in SPLITS:
        if not (data_dir / f"{split}.json").exists():
            print(f"  [SKIP] {data_dir / f'{split}.json'} not found")
            continue
        samples = get_split_samples(data_dir, split)
        print(f"\n  {split}: {len(samples)} samples")

        loader = DataLoader(
            SplitImageDataset(samples), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn,
        )

        columns = {k: [] for k in ("image_emb", "text_emb", "safety_label", "image_tone", "text_tone", "conflict_type")}
        for batch in tqdm(loader, desc=f"    {encoder_name}/{split}"):
            img_emb, txt_emb = encoder.embed(batch["images"], batch["texts"])
            columns["image_emb"].append(img_emb.float().cpu())
            columns["text_emb"].append(txt_emb.float().cpu())
            for key in ("safety_label", "image_tone", "text_tone", "conflict_type"):
                columns[key].append(batch[key])

        cache = {k: torch.cat(v, dim=0) for k, v in columns.items()}
        cache_path = out_dir / f"{split}.pt"
        with atomic_output(cache_path) as tmp_path:
            torch.save(cache, tmp_path)
        print(f"    Saved: {cache_path} ({tuple(cache['image_emb'].shape)})")

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Cache frozen encoder embeddings for PRISM training/evaluation")
    parser.add_argument("--encoders", nargs="+", default=list(ENCODER_REGISTRY), choices=list(ENCODER_REGISTRY))
    parser.add_argument("--data_dir", type=Path, default=None, help="Directory with {train,val,test}.json")
    parser.add_argument("--cache_dir", type=Path, default=None,
                        help="Output directory for <encoder>/{train,val,test}.pt (default: prism_monitor/cache)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (default: per-encoder)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers for parallel image loading")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing cache files of the selected encoders")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    data_dir = args.data_dir or cfg.split_data_dir
    cache_dir = args.cache_dir or cfg.cache_dir
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Cache dir: {cache_dir}")

    existing = [
        path
        for enc_name in args.encoders
        for path in [cache_dir / enc_name / f"{split}.pt" for split in SPLITS] + [cache_dir / enc_name / PROJECTION_FILE]
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise SystemExit(
            f"Error: cache files already exist: {[str(p) for p in existing]}\n"
            "Move them aside, or pass --overwrite to replace them."
        )

    for enc_name in args.encoders:
        bs = args.batch_size or DEFAULT_BATCH_SIZES[enc_name]
        cache_encoder(enc_name, data_dir, cache_dir, device, batch_size=bs, num_workers=args.num_workers)

    print(f"\nAll caches saved to: {cache_dir}")


if __name__ == "__main__":
    main()
