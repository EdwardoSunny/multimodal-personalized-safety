"""Frozen encoders that turn (image, query) pairs into PRISM input embeddings.

PRISM never runs the encoder inside the trained model: embeddings are computed
once with `Encoder.embed()` (see `cache_embeddings.py`) and the monitor head is
trained on them. Inference uses the same `embed()` path.

Image loading and preprocessing follow the research code that computed the paper's
embedding caches: `load_image` only downscales images above 89M pixels, every encoder's
own processor does the rest, and Qwen3-VL first resizes to 384x384 (see `Qwen3VLEncoder`).
Recomputing the paper's SigLIP2 test-split embeddings with this code (and Pillow < 13, as
pinned in pyproject.toml; Pillow 13 changed bilinear resizing) reproduces its cached image
embeddings (per-image cosine similarity 0.99999 or higher); text embeddings are numerically
equivalent (max abs difference ~2.7e-5, from batched vs. per-sample GPU kernels), and
`PRISMPredictor("default")` on the paper's test images gives the paper's test AUROC (0.9832).

Every Hub download is pinned to a commit (`revision` / `processor_revision` in
`ENCODER_REGISTRY`), the revisions current when the paper's caches were computed.

Encoders:
- `siglip_so400m`: google/siglip2-large-patch16-512 (1024-d). The key is historical: the
                   encoder is SigLIP2-large, not SigLIP-so400m.
- `clip`:          openai/clip-vit-large-patch14 (768-d)
- `evaclip`:       BAAI/EVA-CLIP-8B (1280-d, CLIP processor). Loads the repo's own
                   modeling code (`trust_remote_code=True`).
- `qwen3vl`:       Qwen/Qwen3-VL-2B-Instruct ViT + LM, mean-pooled and linearly
                   projected to 1152-d. The projection is randomly initialised and
                   frozen; it is saved next to the cache/checkpoint as
                   `encoder_projection.safetensors`.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECTION_FILE = "encoder_projection.safetensors"
# Pillow's decompression-bomb warning threshold; larger images are downscaled to it
MAX_IMAGE_PIXELS = 89_000_000


def load_image(path: str | Path) -> Image.Image:
    """Load an RGB image; images above 89M pixels are downscaled (LANCZOS) to 89M pixels.

    Every other image is passed to the encoder at full size, as in the paper's embedding caches.
    """
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if w * h > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


class Encoder(nn.Module):
    """Base class: frozen backbone + its own preprocessing."""

    embed_dim: int

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def embed(self, images: list[Image.Image], texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return float32 `(image_emb [B, D], text_emb [B, D])`."""
        raise NotImplementedError

    @property
    def projection(self) -> nn.Module | None:
        """Frozen random projection that must be saved with the embeddings (or None)."""
        return None


class _ProcessorEncoder(Encoder):
    """Dual encoders whose HF processor handles image and text together."""

    TEXT_MAX_LENGTH = 64

    def __init__(self, processor_id: str, processor_revision: str):
        super().__init__()
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(processor_id, revision=processor_revision)

    def _freeze(self):
        for param in self._model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def embed(self, images, texts):
        inputs = self._processor(
            text=texts, images=images,
            padding="max_length", truncation=True, max_length=self.TEXT_MAX_LENGTH,
            return_tensors="pt",
        )
        device = self.device
        pixel_values = inputs["pixel_values"].to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).to(device)
        return (
            self.encode_image(pixel_values).float(),
            self.encode_text(input_ids, attention_mask).float(),
        )


class SigLIPEncoder(_ProcessorEncoder):
    def __init__(self, model_id: str, revision: str, processor_id: str, processor_revision: str):
        super().__init__(processor_id, processor_revision)
        from transformers import SiglipModel

        print(f"  Loading SigLIP encoder: {model_id}@{revision}")
        self._model = SiglipModel.from_pretrained(model_id, revision=revision)
        self._freeze()
        self.embed_dim = self._model.config.vision_config.hidden_size

    def encode_image(self, pixel_values):
        return self._model.vision_model(pixel_values=pixel_values).pooler_output

    def encode_text(self, input_ids, attention_mask):
        return self._model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output


class CLIPEncoder(_ProcessorEncoder):
    def __init__(self, model_id: str, revision: str, processor_id: str, processor_revision: str):
        super().__init__(processor_id, processor_revision)
        from transformers import CLIPModel

        print(f"  Loading CLIP encoder: {model_id}@{revision}")
        self._model = CLIPModel.from_pretrained(model_id, revision=revision)
        self._freeze()
        self.embed_dim = self._model.config.projection_dim

    def encode_image(self, pixel_values):
        return self._model.get_image_features(pixel_values=pixel_values)

    def encode_text(self, input_ids, attention_mask):
        return self._model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)


class EVACLIPEncoder(_ProcessorEncoder):
    def __init__(self, model_id: str, revision: str, processor_id: str, processor_revision: str):
        super().__init__(processor_id, processor_revision)
        from transformers import AutoConfig, AutoModel

        # The model class comes from the Hub repo (remote code), so pinning `revision` also pins that code.
        print(f"  Loading EVA-CLIP encoder: {model_id}@{revision}")
        self.embed_dim = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True).projection_dim
        self._model = AutoModel.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, torch_dtype=torch.float16,
        )
        self._freeze()

    def encode_image(self, pixel_values):
        return self._model.encode_image(pixel_values.to(dtype=torch.float16)).float()

    def encode_text(self, input_ids, attention_mask):
        return self._model.encode_text(input_ids, attention_mask=attention_mask).float()


class Qwen3VLEncoder(Encoder):
    """Qwen3-VL ViT for images and LM backbone for text.

    Image: resize to 384x384 → Qwen image processor (default dynamic resolution,
           144 vision tokens for 384x384) → ViT → mean pool → project
    Text:  Qwen tokenizer (max 128 tokens) → LM → masked mean pool → project

    The 384x384 step reproduces the paper's pipeline exactly: torchvision `Resize`
    (bilinear), then the [-1, 1]-normalized tensor converted back to a uint8 image.
    """

    TEXT_MAX_LENGTH = 128
    IMAGE_SIZE = 384

    def __init__(self, model_id: str, revision: str, processor_id: str, processor_revision: str,
                 output_dim: int = 1152):
        super().__init__()
        from torchvision import transforms
        from transformers import AutoImageProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        print(f"  Loading Qwen3-VL encoder: {model_id}@{revision}")
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=torch.float16, device_map="cpu",
        )
        self._vision = qwen.model.visual
        self._lm = qwen.model
        hidden = qwen.config.text_config.hidden_size  # ViT output is merged into the LM width
        del qwen.lm_head
        del qwen
        for p in self._vision.parameters():
            p.requires_grad = False
        for p in self._lm.parameters():
            p.requires_grad = False

        self._resize = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self._image_processor = AutoImageProcessor.from_pretrained(processor_id, revision=processor_revision)
        self._tokenizer = AutoTokenizer.from_pretrained(processor_id, revision=processor_revision)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self.embed_dim = output_dim
        self._projection = nn.ModuleDict({
            "vision": nn.Linear(hidden, output_dim),
            "text": nn.Linear(hidden, output_dim),
        })
        for p in self._projection.parameters():
            p.requires_grad = False

    @property
    def projection(self) -> nn.Module:
        return self._projection

    def _resized(self, image: Image.Image) -> Image.Image:
        """384x384 uint8 copy of `image`, through the same [-1, 1] tensor round trip as the paper's code."""
        array = self._resize(image).permute(1, 2, 0).numpy()
        return Image.fromarray(np.clip((array * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8))

    @torch.no_grad()
    def embed(self, images, texts):
        device = self.device

        qwen_inputs = self._image_processor(images=[self._resized(image) for image in images], return_tensors="pt")
        pixel_values = qwen_inputs["pixel_values"].to(device, dtype=torch.float16)
        grid_thw = qwen_inputs["image_grid_thw"].to(device)
        vision_out = self._vision(pixel_values, grid_thw=grid_thw)
        vision_tokens = vision_out[0] if isinstance(vision_out, tuple) else vision_out
        merge = getattr(self._vision, "spatial_merge_size", 2)
        token_counts = [(int(t) * int(h) * int(w)) // (merge * merge) for (t, h, w) in grid_thw.tolist()]
        pooled, offset = [], 0
        for n in token_counts:
            pooled.append(vision_tokens[offset:offset + n].float().mean(dim=0))
            offset += n
        image_emb = self._projection["vision"](torch.stack(pooled, dim=0))

        tokens = self._tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.TEXT_MAX_LENGTH, return_tensors="pt",
        )
        attention_mask = tokens["attention_mask"].to(device)
        hidden = self._lm(input_ids=tokens["input_ids"].to(device), attention_mask=attention_mask).last_hidden_state.float()
        mask = attention_mask.unsqueeze(-1).float()
        text_emb = self._projection["text"]((hidden * mask).sum(1) / mask.sum(1).clamp(min=1))

        return image_emb, text_emb


# Hub commits current when the paper's embedding caches were computed (March 2026)
SIGLIP2_REVISION = "49488218e80259885f3be61d7a9455faf833b7a8"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
EVACLIP_REVISION = "0e4dca944e8ece27eb9dfe4a488c0ed0c4644fc9"
QWEN3VL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"

ENCODER_REGISTRY = {
    "siglip_so400m": {  # historical key; the encoder is SigLIP2-large
        "class": SigLIPEncoder,
        "model_id": "google/siglip2-large-patch16-512",
        "revision": SIGLIP2_REVISION,
        "processor_id": "google/siglip2-large-patch16-512",
        "processor_revision": SIGLIP2_REVISION,
        "embed_dim": 1024,
        "display_name": "SigLIP2 Large",
    },
    "clip": {
        "class": CLIPEncoder,
        "model_id": "openai/clip-vit-large-patch14",
        "revision": CLIP_REVISION,
        "processor_id": "openai/clip-vit-large-patch14",
        "processor_revision": CLIP_REVISION,
        "embed_dim": 768,
        "display_name": "CLIP ViT-L/14",
    },
    "evaclip": {
        "class": EVACLIPEncoder,
        "model_id": "BAAI/EVA-CLIP-8B",
        "revision": EVACLIP_REVISION,
        "processor_id": "openai/clip-vit-large-patch14",
        "processor_revision": CLIP_REVISION,
        "embed_dim": 1280,
        "display_name": "EVA-CLIP 8B",
    },
    "qwen3vl": {
        "class": Qwen3VLEncoder,
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": QWEN3VL_REVISION,
        "processor_id": "Qwen/Qwen3-VL-2B-Instruct",
        "processor_revision": QWEN3VL_REVISION,
        "embed_dim": 1152,
        "display_name": "Qwen3-VL 2B",
        "random_projection": True,
    },
}


def create_encoder(name: str, projection_path: str | Path | None = None) -> Encoder:
    """Instantiate an encoder from `ENCODER_REGISTRY`.

    For encoders with a random projection (Qwen3-VL), pass `projection_path` to load
    the projection that produced a given cache/checkpoint; without it a fresh random
    projection is drawn (only valid when building a new embedding cache).
    """
    info = ENCODER_REGISTRY[name]
    encoder = info["class"](
        model_id=info["model_id"], revision=info["revision"],
        processor_id=info["processor_id"], processor_revision=info["processor_revision"],
    )
    if encoder.projection is not None and projection_path is not None:
        from safetensors.torch import load_file

        encoder.projection.load_state_dict(load_file(str(projection_path)))
    return encoder.eval()
