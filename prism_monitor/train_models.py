#!/usr/bin/env python3
"""Train PRISM Monitor models on cached encoder embeddings.

Run `cache_embeddings.py` first. Each model is written to
`checkpoints/<name>/` as `model.safetensors` + `config.json`, the same format as
the released checkpoints, plus `training_info.json` (best val F1, epochs,
timing and per-epoch history; not needed for loading).

Usage:
    uv run python train_models.py                                  # all four configs
    uv run python train_models.py --models default qwen_default    # selected configs
    uv run python train_models.py --models default --wandb         # log to Weights & Biases

Training into a non-empty `checkpoints/<name>/` (e.g. downloaded released weights) is refused
unless --overwrite is passed; --skip_existing skips configs that already have a model.
"""

import argparse
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from atomic_io import atomic_output, write_json_atomic
from config import PRISM_CONFIGS, get_config
from model.architecture import PRISMMonitor
from model.dataset import create_cached_data_loaders
from model.encoders import ENCODER_REGISTRY, PROJECTION_FILE
from model.loading import MODEL_FILE, checkpoint_config
from model.trainer import Trainer

TRAINING_INFO_FILE = "training_info.json"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model_name, base_config, device, use_wandb, batch_size, num_epochs, learning_rate, warmup_steps, num_workers):
    cfg = PRISM_CONFIGS[model_name]
    checkpoint_dir = base_config.checkpoint_dir / cfg.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Training: {cfg.display_name} ({cfg.name})")
    print(f"  Encoder:  {cfg.encoder}")
    print(f"{'=' * 60}")

    set_seed(base_config.training.seed)

    cache_dir = base_config.cache_dir / cfg.encoder
    if not (cache_dir / "train.pt").exists():
        raise FileNotFoundError(
            f"No cached embeddings in {cache_dir}. Run: uv run python cache_embeddings.py --encoders {cfg.encoder}"
        )
    print(f"  Using cached embeddings from: {cache_dir}")
    train_loader, val_loader, _ = create_cached_data_loaders(
        cache_dir=cache_dir, batch_size=batch_size, num_workers=num_workers,
    )
    model = PRISMMonitor(
        embed_dim=ENCODER_REGISTRY[cfg.encoder]["embed_dim"],
        safety_hidden=base_config.model.safety_hidden,
        conflict_hidden=base_config.model.conflict_hidden,
    )
    ckpt_config = checkpoint_config(cfg, base_config.model.safety_hidden, base_config.model.conflict_hidden)

    if use_wandb:
        import wandb

        wandb.init(
            project=base_config.wandb.project,
            name=cfg.display_name,
            tags=[cfg.encoder],
            config={
                "model_name": cfg.name,
                "display_name": cfg.display_name,
                "encoder": cfg.encoder,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "trainable_params": model.num_trainable_params,
                "total_params": model.num_total_params,
                **base_config.to_dict(),
            },
            reinit=True,
        )

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        learning_rate=learning_rate, weight_decay=base_config.training.weight_decay,
        num_epochs=num_epochs, warmup_steps=warmup_steps,
        patience=base_config.training.patience, min_delta=base_config.training.min_delta,
        safety_weight=base_config.training.safety_weight,
        conflict_weight=base_config.training.conflict_weight,
        img_tone_weight=base_config.training.img_tone_weight,
        txt_tone_weight=base_config.training.txt_tone_weight,
        checkpoint_dir=checkpoint_dir, checkpoint_config=ckpt_config, device=device,
        use_wandb=use_wandb, wandb_log_freq=base_config.wandb.log_freq,
    )

    start = time.time()
    history = trainer.train()
    elapsed = time.time() - start
    if trainer.best_epoch == 0:
        # Nothing was saved, so checkpoint_dir does not hold weights from this run
        raise RuntimeError(f"{cfg.name}: no epoch reached a val F1 above min_delta; no model was saved to {checkpoint_dir}")

    info = {
        "model": cfg.name,
        "best_val_f1": trainer.best_val_f1,
        "best_epoch": trainer.best_epoch,
        "epochs_trained": trainer.current_epoch + 1,
        "trainable_params": model.num_trainable_params,
        "training_time_seconds": elapsed,
    }
    write_json_atomic(checkpoint_dir / TRAINING_INFO_FILE, {**info, "history": history}, indent=2)

    projection = cache_dir / PROJECTION_FILE
    if projection.exists():
        with atomic_output(checkpoint_dir / PROJECTION_FILE) as tmp_path:
            shutil.copy(projection, tmp_path)

    if use_wandb:
        wandb.run.summary["best_val_f1"] = trainer.best_val_f1
        wandb.run.summary["epochs_trained"] = trainer.current_epoch + 1
        wandb.run.summary["training_time_minutes"] = elapsed / 60
        wandb.run.summary["trainable_params"] = model.num_trainable_params
        wandb.finish()

    del model, trainer, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return info


def parse_args():
    parser = argparse.ArgumentParser(description="Train PRISM Monitor models on cached embeddings")
    parser.add_argument("--models", nargs="+", choices=list(PRISM_CONFIGS) + ["all"], default=["all"],
                        metavar="NAME", help=f"Configs to train (default: all): {', '.join(PRISM_CONFIGS)}")
    parser.add_argument("--skip_existing", action="store_true", help="Skip configs that already have a checkpoint")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow training into non-empty checkpoint directories (their files are replaced)")
    parser.add_argument("--checkpoint_dir", type=Path, default=None,
                        help="Output directory for <config>/ checkpoints (default: prism_monitor/checkpoints)")
    parser.add_argument("--cache_dir", type=Path, default=None,
                        help="Embedding cache from cache_embeddings.py (default: prism_monitor/cache)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (entity from WANDB_ENTITY)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir

    if args.seed is not None:
        cfg.training.seed = args.seed
    set_seed(cfg.training.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    selected = list(PRISM_CONFIGS) if "all" in args.models else args.models
    batch_size = args.batch_size if args.batch_size is not None else cfg.training.batch_size
    num_epochs = args.epochs if args.epochs is not None else cfg.training.num_epochs
    learning_rate = args.lr if args.lr is not None else cfg.training.learning_rate
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else cfg.training.warmup_steps

    to_train = []
    for model_name in selected:
        if args.skip_existing and (cfg.checkpoint_dir / model_name / MODEL_FILE).exists():
            print(f"\nSkipping {model_name} — checkpoint exists")
            continue
        to_train.append(model_name)
    occupied = [
        str(cfg.checkpoint_dir / name) for name in to_train
        if (cfg.checkpoint_dir / name).is_dir() and any((cfg.checkpoint_dir / name).iterdir())
    ]
    if occupied and not args.overwrite:
        raise SystemExit(
            f"Error: checkpoint directories are not empty: {occupied}\n"
            "They may hold downloaded released weights. Choose another --checkpoint_dir, pass --skip_existing, "
            "or pass --overwrite to replace their files."
        )

    results = []
    for model_name in to_train:
        results.append(train_model(model_name, cfg, device, args.wandb, batch_size, num_epochs,
                                   learning_rate, warmup_steps, args.num_workers))

    print(f"\n{'=' * 60}")
    for r in results:
        print(f"  {r['model']:<20s} best val F1 {r['best_val_f1']:.4f} (epoch {r['best_epoch']}/{r['epochs_trained']})")
    print(f"Checkpoints in: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
