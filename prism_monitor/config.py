"""PRISM model configurations, training hyperparameters and default paths."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PRISMConfig:
    """A PRISM model: the Bi-FiLM monitor head on one frozen encoder."""

    name: str
    display_name: str
    encoder: str  # key of model.encoders.ENCODER_REGISTRY


# `default`, `clip_default` and `evaclip_default` are released on the Hugging Face Hub;
# `qwen_default` is trained from scratch (see README).
PRISM_CONFIGS: dict[str, PRISMConfig] = {
    cfg.name: cfg
    for cfg in (
        PRISMConfig("default", "PRISM (SigLIP2)", "siglip_so400m"),
        PRISMConfig("clip_default", "PRISM (CLIP)", "clip"),
        PRISMConfig("evaclip_default", "PRISM (EVA-CLIP)", "evaclip"),
        PRISMConfig("qwen_default", "PRISM (Qwen3-VL)", "qwen3vl"),
    )
}


@dataclass
class ModelConfig:
    safety_hidden: int = 512
    conflict_hidden: int = 128


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 15
    warmup_steps: int = 20
    patience: int = 5
    min_delta: float = 0.001
    seed: int = 42

    # Multi-task loss weights
    safety_weight: float = 1.0
    conflict_weight: float = 0.3
    img_tone_weight: float = 0.1
    txt_tone_weight: float = 0.1


@dataclass
class WandbConfig:
    """Weights & Biases logging (opt-in via `train_models.py --wandb`).

    The entity is taken from the `WANDB_ENTITY` environment variable if set.
    """

    project: str = "prism-monitor"
    log_freq: int = 10


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    split_data_dir: Path = PACKAGE_DIR / "dataset" / "splits"
    cache_dir: Path = PACKAGE_DIR / "cache"
    checkpoint_dir: Path = PACKAGE_DIR / "checkpoints"

    def to_dict(self) -> dict:
        return {"model": asdict(self.model), "training": asdict(self.training)}


def get_config() -> Config:
    return Config()
