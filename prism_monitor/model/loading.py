"""Save/load PRISM checkpoints (`model.safetensors` + `config.json`).

The same format is written by the trainer and published on the Hugging Face Hub
(`scripts/download_checkpoints.py` places released checkpoints under
`prism_monitor/checkpoints/<config_name>/`).

config.json holds what is needed to rebuild the model and its encoder:
- `model`, `display_name`: the PRISM config (see `config.PRISM_CONFIGS`)
- `architecture`: `{"embed_dim", "safety_hidden", "conflict_hidden"}` (`PRISMMonitor` arguments)
- `encoder`: `{"name", "hf_model_id", "processor_id"}` (`name` is a key of `ENCODER_REGISTRY`)

Qwen3-VL checkpoints also contain `encoder_projection.safetensors`, the frozen random
2048→1152 projection their embeddings were cached with (`cache_embeddings.py` writes it
and `train_models.py` copies it into the checkpoint).
"""

import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from atomic_io import atomic_output, write_json_atomic
from config import PRISMConfig
from model.architecture import PRISMMonitor
from model.encoders import ENCODER_REGISTRY, PROJECTION_FILE, create_encoder

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
MODEL_FILE = "model.safetensors"
CONFIG_FILE = "config.json"


def checkpoint_config(cfg: PRISMConfig, safety_hidden: int, conflict_hidden: int) -> dict:
    """config.json contents for a PRISM config and its head sizes."""
    info = ENCODER_REGISTRY[cfg.encoder]
    return {
        "model": cfg.name,
        "display_name": cfg.display_name,
        "architecture": {
            "embed_dim": info["embed_dim"],
            "safety_hidden": safety_hidden,
            "conflict_hidden": conflict_hidden,
        },
        "encoder": {
            "name": cfg.encoder,
            "hf_model_id": info["model_id"],
            "processor_id": info["processor_id"],
        },
    }


def save_checkpoint(model: PRISMMonitor, config: dict, checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    with atomic_output(checkpoint_dir / MODEL_FILE) as tmp_path:
        save_file(state, str(tmp_path))
    write_json_atomic(checkpoint_dir / CONFIG_FILE, config, indent=2)


def resolve_checkpoint_dir(checkpoint: str | Path) -> Path:
    """A config name (e.g. `default`) → `prism_monitor/checkpoints/<name>`; a path is used as-is."""
    path = Path(checkpoint)
    if (path / CONFIG_FILE).exists():
        return path
    named = CHECKPOINT_DIR / str(checkpoint)
    if (named / CONFIG_FILE).exists():
        return named
    raise FileNotFoundError(
        f"No PRISM checkpoint at '{checkpoint}' or '{named}'. "
        "Run `python scripts/download_checkpoints.py` or train with `train_models.py`."
    )


def load_prism(checkpoint: str | Path = "default", device: str | torch.device = "cpu") -> tuple[PRISMMonitor, dict]:
    """Rebuild a PRISM model from `config.json` and load `model.safetensors`.

    Args:
        checkpoint: config name under `prism_monitor/checkpoints/` or a checkpoint directory.
        device: device to place the model on.

    Returns:
        `(model, config)` with the model in eval mode. The model consumes
        `image_emb` / `text_emb` from the matching encoder (`config["encoder"]`).
    """
    ckpt_dir = resolve_checkpoint_dir(checkpoint)
    with open(ckpt_dir / CONFIG_FILE) as f:
        config = json.load(f)
    model = PRISMMonitor(**config["architecture"])
    model.load_state_dict(load_file(str(ckpt_dir / MODEL_FILE)))
    config["checkpoint_dir"] = str(ckpt_dir)
    return model.to(device).eval(), config


class PRISMPredictor:
    """End-to-end inference: (PIL image, query) pairs → deferral probability."""

    def __init__(self, checkpoint: str | Path = "default", device: str | torch.device | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.config = load_prism(checkpoint, self.device)
        name = self.config["encoder"]["name"]
        projection = Path(self.config["checkpoint_dir"]) / PROJECTION_FILE
        if ENCODER_REGISTRY[name].get("random_projection"):
            if not projection.exists():
                raise FileNotFoundError(
                    f"{name} checkpoints need the frozen encoder projection their embeddings were cached with "
                    f"({projection}). cache_embeddings.py writes it and train_models.py copies it into the checkpoint."
                )
            self.encoder = create_encoder(name, projection)
        else:
            self.encoder = create_encoder(name)
        self.encoder.to(self.device)

    @torch.no_grad()
    def predict(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        """Probability `[B]` that each query needs the user's background (i.e. should be deferred)."""
        image_emb, text_emb = self.encoder.embed(images, texts)
        logits = self.model(image_emb=image_emb, text_emb=text_emb)["safety_logits"].squeeze(-1)
        return torch.sigmoid(logits).cpu()
