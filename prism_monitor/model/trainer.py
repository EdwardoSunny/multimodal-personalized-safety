"""
Trainer for PRISM Monitor.

Multi-task learning: BCE safety loss + CrossEntropy aux losses (tone, conflict).
Linear warmup + cosine decay, early stopping on val F1.
"""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import LABEL_KEYS
from model.loading import save_checkpoint

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 15,
        warmup_steps: int = 20,
        patience: int = 5,
        min_delta: float = 0.001,
        # Multi-task loss weights
        safety_weight: float = 1.0,
        conflict_weight: float = 0.3,
        img_tone_weight: float = 0.1,
        txt_tone_weight: float = 0.1,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_config: Optional[dict] = None,
        device: Optional[torch.device] = None,
        use_wandb: bool = False,
        wandb_log_freq: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        print(f"Training on: {self.device}")

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_log_freq = wandb_log_freq

        # Loss weights
        self.loss_weights = {
            "safety": safety_weight,
            "conflict": conflict_weight,
            "img_tone": img_tone_weight,
            "txt_tone": txt_tone_weight,
        }

        # Loss functions
        self.safety_criterion = nn.BCEWithLogitsLoss()
        self.conflict_criterion = nn.CrossEntropyLoss()
        self.tone_criterion = nn.CrossEntropyLoss()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        print(f"Trainable parameters: {model.num_trainable_params:,}")
        print(f"Total parameters: {model.num_total_params:,}")

        total_steps = len(train_loader) * num_epochs
        self.scheduler = self._create_scheduler(warmup_steps, total_steps)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_config = checkpoint_config or {}
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_f1 = 0.0
        self.best_epoch = 0  # 1-based epoch of the saved checkpoint
        self.epochs_without_improvement = 0
        self.training_history = []

    def _create_scheduler(self, warmup_steps: int, total_steps: int) -> LambdaLR:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return LambdaLR(self.optimizer, lr_lambda)

    def compute_loss(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        # Safety loss (always)
        safety_loss = self.safety_criterion(outputs["safety_logits"].squeeze(-1), batch["safety_label"])
        total_loss = self.loss_weights["safety"] * safety_loss
        loss_dict = {"safety": safety_loss.item()}

        # Aux: conflict
        if self.loss_weights["conflict"] > 0:
            conflict_loss = self.conflict_criterion(outputs["conflict_logits"], batch["conflict_type"])
            total_loss = total_loss + self.loss_weights["conflict"] * conflict_loss
            loss_dict["conflict"] = conflict_loss.item()

        # Aux: image tone
        if self.loss_weights["img_tone"] > 0:
            img_tone_loss = self.tone_criterion(outputs["image_tone_logits"], batch["image_tone"])
            total_loss = total_loss + self.loss_weights["img_tone"] * img_tone_loss
            loss_dict["img_tone"] = img_tone_loss.item()

        # Aux: text tone
        if self.loss_weights["txt_tone"] > 0:
            txt_tone_loss = self.tone_criterion(outputs["text_tone_logits"], batch["text_tone"])
            total_loss = total_loss + self.loss_weights["txt_tone"] * txt_tone_loss
            loss_dict["txt_tone"] = txt_tone_loss.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_losses = {"total": 0.0, "safety": 0.0}
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Pass non-label keys (image_emb, text_emb) to the model
            model_inputs = {k: v for k, v in batch.items() if k not in LABEL_KEYS}
            outputs = self.model(**model_inputs)

            loss, loss_dict = self.compute_loss(outputs, batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            num_batches += 1
            self.global_step += 1

            progress_bar.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                safety=f"{loss_dict['safety']:.4f}",
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
            )

            if self.use_wandb and (self.global_step % self.wandb_log_freq == 0):
                log_data = {f"train/{k}_loss": v for k, v in loss_dict.items()}
                log_data["train/learning_rate"] = self.scheduler.get_last_lr()[0]
                log_data["train/global_step"] = self.global_step
                wandb.log(log_data, step=self.global_step)

        return {k: v / num_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        total_losses = {"total": 0.0, "safety": 0.0}
        num_batches = 0
        all_preds, all_labels = [], []

        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            model_inputs = {k: v for k, v in batch.items() if k not in LABEL_KEYS}
            outputs = self.model(**model_inputs)

            _, loss_dict = self.compute_loss(outputs, batch)
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            num_batches += 1

            probs = torch.sigmoid(outputs["safety_logits"].squeeze(-1))
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["safety_label"].long().cpu().tolist())

        metrics = self._calculate_metrics(all_preds, all_labels)
        metrics["loss"] = total_losses["total"] / num_batches
        metrics["safety_loss"] = total_losses["safety"] / num_batches
        return metrics

    def _calculate_metrics(self, preds, labels):
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        accuracy = (tp + tn) / max(1, len(preds))
        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

    def save_best(self):
        """Write the current model as `model.safetensors` + `config.json` (see model.loading)."""
        self.best_epoch = self.current_epoch + 1
        if self.checkpoint_dir is None:
            return
        save_checkpoint(self.model, self.checkpoint_config, self.checkpoint_dir)
        print(f"  Saved best model (F1={self.best_val_f1:.4f})")

    def train(self) -> list[dict]:
        print(f"\nStarting training for {self.num_epochs} epochs")
        if self.val_loader is not None:
            print(f"Early stopping patience: {self.patience}")
        print(f"Loss weights: safety={self.loss_weights['safety']}, conflict={self.loss_weights['conflict']}, "
              f"img_tone={self.loss_weights['img_tone']}, txt_tone={self.loss_weights['txt_tone']}")
        print("-" * 50)

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            train_losses = self.train_epoch()

            epoch_log = {"epoch": epoch + 1, "train": train_losses}
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f} (safety: {train_losses['safety']:.4f})")

            if self.val_loader is not None:
                val_metrics = self.validate()
                epoch_log["val"] = val_metrics

                print(f"  Val Loss: {val_metrics['loss']:.4f} (safety: {val_metrics['safety_loss']:.4f})")
                print(f"  Val F1: {val_metrics['f1']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

                if self.use_wandb:
                    epoch_log_wandb = {
                        "epoch": epoch + 1,
                        # Train epoch averages
                        **{f"train/epoch_{k}_loss": v for k, v in train_losses.items()},
                        # Val losses
                        "val/total_loss": val_metrics["loss"],
                        "val/safety_loss": val_metrics["safety_loss"],
                        # Val classification metrics
                        "val/f1": val_metrics["f1"],
                        "val/accuracy": val_metrics["accuracy"],
                        "val/precision": val_metrics["precision"],
                        "val/recall": val_metrics["recall"],
                        # Best tracking
                        "val/best_f1": self.best_val_f1,
                    }
                    wandb.log(epoch_log_wandb, step=self.global_step)

                if val_metrics["f1"] > self.best_val_f1 + self.min_delta:
                    self.best_val_f1 = val_metrics["f1"]
                    self.epochs_without_improvement = 0
                    print(f"  New best F1: {self.best_val_f1:.4f}")
                    self.save_best()
                else:
                    self.epochs_without_improvement += 1
            elif epoch == self.num_epochs - 1:
                self.save_best()

            # Record the epoch before a possible early stop so the last one is kept.
            self.training_history.append(epoch_log)

            if self.val_loader is not None and self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

        print(f"\nTraining complete. Best Val F1: {self.best_val_f1:.4f}")

        return self.training_history
