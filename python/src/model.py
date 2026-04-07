"""
Text -> Face animation model (v3).

A one-shot feed-forward transformer that maps text to a sequence of
61-channel ARKit blendshape coefficients. One forward pass produces the
full sequence -- no autoregression, no diffusion -- so inference is fast
enough for realtime use in Unreal / LiveLink.

Key differences vs v1:
    1. Text conditioning is now per-frame CROSS-ATTENTION over the full
       CLIP token sequence. v1 squashed CLIP into a single pooled [EOS]
       vector and broadcast-added it to every frame -- so every frame
       received identical text info. Now each frame query attends to the
       specific words it needs (e.g. frame 10 may attend to "smiles",
       frame 60 may attend to "then frowns").
    2. Transformer backbone switched from encoder (self-attn only) to
       decoder (self-attn on frames + cross-attn to text tokens + FFN).
    3. text_proj gained a final LayerNorm for stable cross-attention.
    4. Dropout default raised from 0.1 -> 0.2 to fight overfitting.

Forward API is UNCHANGED: model(texts, n_frames=T) -> [B, T, 61].
sample.py and server.py do not need any modification.

Architecture:

    text prompt
       |
       v
    CLIPTokenizer + CLIPTextModel (frozen)
       |  -> last_hidden_state [B, L, 512], attention_mask [B, L]
       v
    text_proj (Linear -> GELU -> Linear -> LayerNorm)
       |  -> [B, L, latent_dim]
       v
    Learned frame queries [1, T, latent_dim]
       |                +-- self-attention over frames (bidirectional)
       v                +-- cross-attention over text tokens (padding-masked)
    TransformerDecoder (n_layers, n_heads, ff_dim, GELU, pre-norm)
       |  -> [B, T, latent_dim]
       v
    LayerNorm + Linear(latent_dim, 61)
       |  -> [B, T, 61]
       v
    output (normalized blendshape space; denormalize with saved mean/std)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

N_CHANNELS = 61
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # text tower = 512-D features


class TextToFace(nn.Module):
    def __init__(
        self,
        max_frames: int = 240,
        n_channels: int = N_CHANNELS,
        latent_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.2,
        clip_model_name: str = CLIP_MODEL_NAME,
    ):
        super().__init__()
        self.max_frames = max_frames
        self.n_channels = n_channels
        self.latent_dim = latent_dim

        # --- Frozen CLIP text tower (HuggingFace) ---
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip_text = CLIPTextModel.from_pretrained(clip_model_name)
        for p in self.clip_text.parameters():
            p.requires_grad = False
        self.clip_text.eval()

        clip_dim = self.clip_text.config.hidden_size  # 512 for ViT-B/32

        # --- Trainable components ---
        # Project each CLIP token hidden state into our latent space.
        # The final LayerNorm keeps cross-attention keys/values in a
        # consistent scale regardless of the prompt length.
        self.text_proj = nn.Sequential(
            nn.Linear(clip_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Learned per-frame query embeddings (one slot per possible
        # output frame position). These are the "tgt" of the decoder.
        # Small init so they don't dominate the signal early in training.
        self.frame_queries = nn.Parameter(
            torch.randn(1, max_frames, latent_dim) * 0.02
        )

        # Transformer decoder: self-attn on frame queries + cross-attn
        # to text tokens + FFN, repeated n_layers times. Pre-norm is
        # important for training stability with GELU activation.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )

        # Output head: project each timestep token -> 61 blendshape values
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, n_channels)

        # Small init on the output so the model starts near zero (which
        # equals the dataset mean after denormalization). This stabilizes
        # very early training.
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)

    # ----------------------------------------------------------------------
    # Parameter count helpers
    # ----------------------------------------------------------------------

    def trainable_parameters(self):
        """Return only the parameters we actually train (excludes frozen CLIP)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ----------------------------------------------------------------------
    # Text -> CLIP embedding (full token sequence, not just pooled)
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize + run CLIP text encoder.

        Returns:
            hidden_states:  [B, L, clip_dim] per-token features
            attention_mask: [B, L] long, 1 = valid token, 0 = padding
        """
        device = next(self.clip_text.parameters()).device
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,          # CLIP's native max token length
            return_tensors="pt",
        ).to(device)
        out = self.clip_text(
            input_ids=tok.input_ids,
            attention_mask=tok.attention_mask,
        )
        return out.last_hidden_state, tok.attention_mask

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------

    def forward(
        self,
        texts: Optional[List[str]] = None,
        n_frames: Optional[int] = None,
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            texts:         list of B strings (or None if text_features provided)
            n_frames:      number of frames to generate; defaults to max_frames
            text_features: pre-computed [B, L, clip_dim] CLIP hidden states
                           (optional, for server-side CLIP caching)
            text_mask:     [B, L] attention mask, required if text_features given
        Returns:
            [B, n_frames, n_channels] in normalized blendshape space
        """
        if n_frames is None:
            n_frames = self.max_frames
        assert n_frames <= self.max_frames, (
            f"n_frames={n_frames} exceeds max_frames={self.max_frames}"
        )

        # (1) Text encoding
        if text_features is None:
            assert texts is not None, "Provide either texts or text_features"
            text_features, text_mask = self.encode_text(texts)
        else:
            assert text_mask is not None, (
                "text_mask must be provided alongside text_features"
            )
            assert text_features.dim() == 3

        B = text_features.shape[0]
        text_emb = self.text_proj(text_features)  # [B, L, latent_dim]

        # (2) Build frame queries: take first n_frames positional slots and
        #     broadcast across the batch.
        queries = self.frame_queries[:, :n_frames, :]           # [1, T, D]
        queries = queries.expand(B, -1, -1).contiguous()        # [B, T, D]

        # (3) Decoder. memory_key_padding_mask uses the convention
        #     True = ignore this key position, so we invert the CLIP mask.
        memory_key_padding_mask = (text_mask == 0)              # [B, L]
        x = self.transformer(
            tgt=queries,
            memory=text_emb,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, T, D]

        # (4) Output head
        x = self.out_norm(x)
        out = self.out_proj(x)                                  # [B, T, n_channels]
        return out
