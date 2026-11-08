"""PRISM Monitor architecture.

Frozen-encoder embeddings (see `model.encoders`) → Bi-FiLM fusion →
modality gate → multi-task heads. The trainable model only sees embeddings.
"""

import torch
import torch.nn as nn


class BiFiLMFusion(nn.Module):
    """
    Bi-Directional Feature-wise Linear Modulation.

    V-Stream: text modulates image  →  v' = (1 + Δγ_v) * v + Δβ_v
    T-Stream: image modulates text  →  t' = (1 + Δγ_t) * t + Δβ_t

    Output: cat([v', t']) → [B, 2 * embed_dim]
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.text_to_film = nn.Linear(embed_dim, embed_dim * 2)
        self.image_to_film = nn.Linear(embed_dim, embed_dim * 2)

        for lin in (self.text_to_film, self.image_to_film):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        dg_v, db_v = self.text_to_film(text_features).chunk(2, dim=-1)
        v_mod = (1.0 + dg_v) * image_features + db_v

        dg_t, db_t = self.image_to_film(image_features).chunk(2, dim=-1)
        t_mod = (1.0 + dg_t) * text_features + db_t

        return torch.cat([v_mod, t_mod], dim=-1)


class PRISMMonitor(nn.Module):
    """
    PRISM Monitor: multimodal binary deferral classifier.

    Architecture:
        1. Aux tone heads      →  image_tone, text_tone [B, 3]
        2. BiFiLM fusion       →  fused [B, 2D]
        3. Modality gate       →  gated [B, 2D]
        4. Aux conflict head   →  conflict [B, 3]
        5. Safety head         →  logit [B, 1]
           input: gated_fused [2D] + raw_text [D] = [3D]
    """

    def __init__(
        self,
        embed_dim: int,
        safety_hidden: int = 512,
        conflict_hidden: int = 128,
        num_tones: int = 3,
        num_conflict_types: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fused_dim = embed_dim * 2

        # Auxiliary: unimodal tone heads
        self.img_tone_head = nn.Linear(embed_dim, num_tones)
        self.txt_tone_head = nn.Linear(embed_dim, num_tones)

        # Fusion
        self.bifilm = BiFiLMFusion(embed_dim)

        # Modality gate
        self.modality_gate = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(self.fused_dim)

        # Auxiliary: conflict head
        self.conflict_head = nn.Sequential(
            nn.Linear(self.fused_dim, conflict_hidden),
            nn.ReLU(),
            nn.Linear(conflict_hidden, num_conflict_types),
        )

        # Safety head: fused [2D] + raw text [D] = [3D]
        self.safety_head = nn.Sequential(
            nn.Linear(self.fused_dim + embed_dim, safety_hidden),
            nn.ReLU(),
            nn.Linear(safety_hidden, 1),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        # Aux: unimodal tone predictions
        image_tone_logits = self.img_tone_head(image_emb)
        text_tone_logits = self.txt_tone_head(text_emb)

        # Fusion + gating
        fused = self.bifilm(image_emb, text_emb)
        gate = self.modality_gate(fused)
        fused = self.fusion_norm(gate * fused)

        # Aux: conflict prediction
        conflict_logits = self.conflict_head(fused)

        # Safety: fused + raw text residual
        safety_logits = self.safety_head(torch.cat([fused, text_emb], dim=-1))

        return {
            "safety_logits": safety_logits,
            "image_tone_logits": image_tone_logits,
            "text_tone_logits": text_tone_logits,
            "conflict_logits": conflict_logits,
            "fused_features": fused,
        }

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
