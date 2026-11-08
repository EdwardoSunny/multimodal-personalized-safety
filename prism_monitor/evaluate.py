#!/usr/bin/env python3
"""Evaluate PRISM Monitor checkpoints on the test split (cached embeddings).

Loads each checkpoint with `model.loading.load_prism` and reports AUROC, AUPRC,
F1, accuracy, ECE, Brier score and FNR at 30%/50% deferral budgets.

Usage:
    uv run python evaluate.py                                   # all configs with a checkpoint
    uv run python evaluate.py --models default evaclip_default  # selected configs

Refuses to replace an existing `<output_dir>/eval_results.json` unless --overwrite is passed.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from atomic_io import write_json_atomic
from config import PRISM_CONFIGS, get_config
from model.dataset import LABEL_KEYS, CachedEmbeddingDataset
from model.loading import CONFIG_FILE, load_prism

OUTPUT_DIR = Path(__file__).resolve().parent / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(y_true, y_probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_probs > bin_boundaries[i]) & (y_probs <= bin_boundaries[i + 1])
        if np.any(in_bin):
            ece += np.abs(np.mean(y_true[in_bin]) - np.mean(y_probs[in_bin])) * np.mean(in_bin)
    return float(ece)


def compute_fnr_at_budget(y_true, y_scores, budget=0.3):
    """False-negative rate when deferring the top `budget` fraction of queries by score."""
    n_defer = int(len(y_true) * budget)
    threshold = np.sort(y_scores)[::-1][min(n_defer, len(y_scores) - 1)]
    not_deferred = y_scores < threshold
    total_positives = np.sum(y_true == 1)
    if total_positives == 0:
        return 0.0
    fn = np.sum((y_true == 1) & not_deferred)
    return float(fn / total_positives)


def compute_metrics(y_true, y_scores, threshold=0.5):
    preds = (y_scores >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_scores)),
        "auprc": float(average_precision_score(y_true, y_scores)),
        "f1": float(f1_score(y_true, preds)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "ece": compute_ece(y_true, y_scores),
        "brier": float(brier_score_loss(y_true, y_scores)),
        "fnr_at_30": compute_fnr_at_budget(y_true, y_scores, 0.3),
        "fnr_at_50": compute_fnr_at_budget(y_true, y_scores, 0.5),
        "n_samples": len(y_true),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate a single model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(checkpoint_dir: Path, cache_dir: Path, device):
    """Evaluate one checkpoint on `<cache_dir>/<encoder>/test.pt`."""
    model, config = load_prism(checkpoint_dir, device)
    encoder = config["encoder"]["name"]
    print(f"\n  Evaluating: {config['display_name']} ({config['model']})")

    test_ds = CachedEmbeddingDataset(cache_dir / encoder / "test.pt")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True)

    scores, labels = [], []
    for batch in tqdm(test_loader, desc=f"    {config['model']}", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        model_inputs = {k: v for k, v in batch.items() if k not in LABEL_KEYS}
        with torch.no_grad():
            probs = torch.sigmoid(model(**model_inputs)["safety_logits"].squeeze(-1))
        scores.extend(probs.cpu().tolist())
        labels.extend(batch["safety_label"].long().cpu().tolist())

    metrics = {
        "model": config["model"],
        "display_name": config["display_name"],
        "encoder": encoder,
        **compute_metrics(np.array(labels, dtype=np.float64), np.array(scores, dtype=np.float64)),
        "trainable_params": model.num_trainable_params,
    }

    print(
        f"    AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
        f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
        f"ECE={metrics['ece']:.4f}  Brier={metrics['brier']:.4f}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PRISM Monitor checkpoints on the test split")
    parser.add_argument("--models", nargs="+", default=["all"], choices=list(PRISM_CONFIGS) + ["all"], metavar="NAME",
                        help="Configs to evaluate (default: all with a checkpoint)")
    parser.add_argument("--checkpoint_dir", type=Path, default=None,
                        help="Directory containing <config>/model.safetensors (default: prism_monitor/checkpoints)")
    parser.add_argument("--cache_dir", type=Path, default=None,
                        help="Embedding cache from cache_embeddings.py (default: prism_monitor/cache)")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing <output_dir>/eval_results.json")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    checkpoint_root = args.checkpoint_dir or cfg.checkpoint_dir
    cache_dir = args.cache_dir or cfg.cache_dir
    results_path = args.output_dir / "eval_results.json"
    if results_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Error: {results_path} already exists. Choose another --output_dir, move it aside, "
            "or pass --overwrite to replace it."
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model_names = list(PRISM_CONFIGS) if "all" in args.models else args.models
    available = []
    for name in model_names:
        if (checkpoint_root / name / CONFIG_FILE).exists():
            available.append(name)
        else:
            print(f"  [SKIP] No checkpoint for {name}")

    if not available:
        print("\nNo checkpoints found. Download them (scripts/download_checkpoints.py) or run train_models.py.")
        return

    print(f"\nEvaluating {len(available)} model(s)")
    results = [evaluate_model(checkpoint_root / name, cache_dir, device) for name in available]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(results_path, results, indent=2)

    print(f"\n{'=' * 90}")
    print(f"  {'Model':<25s} {'AUROC':>7s} {'AUPRC':>7s} {'F1':>7s} {'Acc':>7s} {'ECE':>7s} {'Brier':>7s} {'FNR@30':>7s}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for r in results:
        print(
            f"  {r['display_name']:<25s} "
            f"{r['auroc']:>7.4f} {r['auprc']:>7.4f} {r['f1']:>7.4f} "
            f"{r['accuracy']:>7.4f} {r['ece']:>7.4f} {r['brier']:>7.4f} {r['fnr_at_30']:>7.4f}"
        )
    print(f"{'=' * 90}")

    best = max(results, key=lambda r: r["auroc"])
    print(f"\n  Best AUROC: {best['display_name']} = {best['auroc']:.4f}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
