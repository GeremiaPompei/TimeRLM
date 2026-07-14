"""ITFormerModel — instruction-aware time series LLM implementing BaseModel.

Data flow:
    time_series (B,L,V)  →  Chronos2TSEncoder  →  (B,V,P,d_model)
        →  ts_project  →  (B,V,P,it_d_model)
    query_ids  →  LLM.embed  →  query_project  →  (B,Q,it_d_model)
        →  ITFormerFusion(query, ts, stage)  →  (B,prefix_num,it_d_model)
            →  fusion_project  →  (B,prefix_num,llm_d_model)
                →  merge into input_ids embeddings at <|image_pad|> positions
                    →  LLM forward  →  loss

Ported from ITFormer-clone/models/ITFormer.py.
"""

from __future__ import annotations

import math
import os
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.base import BaseModel
from src.models.chronos2_encoder import Chronos2TSEncoder
from src.models.qwen_backbone import QwenBackbone

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Positional encoding primitives
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for time dimension (stages 1-2)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, 1, max_len, d_model)
        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Return positional encoding slice matching x's sequence length.

        Args:
            x: Tensor whose dim-1 length determines the slice size.
            offset: Starting position in the encoding table.

        Returns:
            Positional encoding tensor broadcastable to x.
        """
        return self.pe[0, :, offset : offset + x.size(1), :]


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding for cross-cycle analysis (stages 3-4)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RotaryPositionalEncoding."

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        dim = torch.arange(0, d_model // 2, dtype=torch.float)
        div_term = torch.exp(dim * -(math.log(10000.0) / (d_model // 2)))

        angle = position * div_term
        sin_part = torch.sin(angle)
        cos_part = torch.cos(angle)

        # Shape: (1, 1, max_len, d_model)
        pe = torch.cat([sin_part, cos_part], dim=-1).unsqueeze(0).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary positional encoding to *x*.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            offset: Starting position.

        Returns:
            Tensor with rotary encoding applied, same shape as *x*.
        """
        seq_len = x.size(1)
        pe = self.pe[0, :, offset : offset + seq_len, :]
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat(
            [
                x1 * pe[..., :half] - x2 * pe[..., half:],
                x1 * pe[..., half:] + x2 * pe[..., :half],
            ],
            dim=-1,
        )


class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embedding for the variable dimension."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, 1, max_len, d_model))

        # Initialise with sinusoidal values
        pe_init = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe_init[:, 0::2] = torch.sin(position * div_term)
        pe_init[:, 1::2] = torch.cos(position * div_term)
        self.pe.data.copy_(pe_init.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return self.pe[0, :, offset : offset + x.size(1), :]


# ---------------------------------------------------------------------------
# Stochastic depth
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularisation."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep_prob).floor_()
        return x.div(keep_prob) * mask


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------


class InstructTimeAttention(nn.Module):
    """Two-stage instruction-aware time-series attention.

    Stage 1: Channel attention — aggregate across variables guided by query.
    Stage 2: Time attention — attend over temporal dimension.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # Channel attention projections
        self.channel_query_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.channel_key_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.channel_value_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Time attention projections
        self.query_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, L_q, d) — instruction / prefix tokens.
            memory: (B, L_p, V, d) — encoded time series patches x variables.

        Returns:
            (B, L_q, d) — attended output.
        """
        B, L_q, d = query.size()
        _, L_p, V, _ = memory.size()

        # --- Stage 1: Channel attention guided by query ---
        q_ch = self.channel_query_proj(query)  # (B, L_q, d)
        k_ch = self.channel_key_proj(memory).mean(dim=1)  # (B, V, d)
        v_ch = self.channel_value_proj(memory)  # (B, L_p, V, d)

        q_ch = self.q_norm(q_ch.view(B, self.num_heads, L_q, self.head_dim))
        k_ch = self.k_norm(k_ch.view(B, self.num_heads, V, self.head_dim))
        v_ch = v_ch.view(B, self.num_heads, L_p, V, self.head_dim)

        ch_scores = torch.matmul(q_ch, k_ch.transpose(-2, -1)) / self.scale
        ch_weights = F.softmax(ch_scores, dim=-1)  # (B, H, L_q, V)
        ch_out = torch.einsum("bnqv,bnlvd->bnld", ch_weights, v_ch)  # (B, H, L_p, hd)

        memory_agg = ch_out.transpose(1, 2).contiguous().view(B, L_p, d)

        # --- Stage 2: Time attention ---
        Q = self.q_norm(self.query_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2))
        K = self.k_norm(self.key_proj(memory_agg).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2))
        V_proj = self.value_proj(memory_agg).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V_proj)

        out = out.transpose(1, 2).contiguous().view(B, L_q, d)
        return self.proj(out)


class SeqAttention(nn.Module):
    """Self-attention over the sequence (patch) dimension."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Fusion decoder blocks
# ---------------------------------------------------------------------------


class SelfAttBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = SeqAttention(dim, num_heads=num_heads, qkv_bias=True,
                                 attn_drop=drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.attn(self.norm(x)))


class ITAttBlock(nn.Module):
    """InstructTime attention wrapper with pre-norm and drop path."""

    def __init__(self, dim: int, num_heads: int, drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = InstructTimeAttention(dim, num_heads=num_heads, qkv_bias=True,
                                          attn_drop=drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.drop_path(self.attn(x, memory)))


class DecoderBasicBlock(nn.Module):
    """One fusion decoder layer: self-attn → IT-attn (prefix only) → FFN."""

    def __init__(self, dim: int, num_heads: int, prefix_num: int = 10,
                 mlp_ratio: float = 4.0, drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.prefix_num = prefix_num

        self.self_attn = SelfAttBlock(dim, num_heads, drop=drop, drop_path=drop_path)
        self.it_attn = ITAttBlock(dim, num_heads, drop=drop, drop_path=drop_path)

        hidden = int(dim * mlp_ratio)
        self.ff_prefix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
        )
        self.ff_instruct = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention on full sequence (prefix + query)
        x = x + self.self_attn(x)

        prefix = x[:, : self.prefix_num, :]
        x = self.ff_instruct(x) + x

        # IT-attention: only prefix tokens attend to memory
        prefix = prefix + self.it_attn(prefix, memory)
        prefix = prefix + self.ff_prefix(prefix)

        # Reconstruct full sequence
        x = torch.cat([prefix, x[:, self.prefix_num :, :]], dim=1)
        return x


class ITFormerFusion(nn.Module):
    """Instruction-aware fusion of query embeddings and encoded time series.

    Takes projected query embeddings and projected encoder output, applies
    stage-aware positional encodings, and returns fused prefix tokens.

    Args:
        it_d_model: Internal fusion dimension.
        it_n_heads: Number of attention heads.
        it_layers: Number of decoder blocks.
        it_dropout: Dropout rate.
        prefix_num: Number of learnable prefix tokens (output length).
    """

    def __init__(
        self,
        it_d_model: int = 896,
        it_n_heads: int = 16,
        it_layers: int = 2,
        it_dropout: float = 0.1,
        prefix_num: int = 25,
    ):
        super().__init__()
        self.prefix_num = prefix_num
        self.prefix_token = nn.Parameter(torch.randn(1, prefix_num, it_d_model))

        self.layers = nn.ModuleList(
            [
                DecoderBasicBlock(
                    dim=it_d_model,
                    num_heads=it_n_heads,
                    prefix_num=prefix_num,
                    drop=it_dropout,
                )
                for _ in range(it_layers)
            ]
        )
        self.norm = nn.LayerNorm(it_d_model)

        # Positional encodings
        self.time_pos = SinusoidalPositionalEncoding(it_d_model)
        self.var_pos = LearnablePositionalEmbedding(it_d_model)
        self.instruc_pos = SinusoidalPositionalEncoding(it_d_model)
        self.cycle_pos = RotaryPositionalEncoding(it_d_model)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        stage: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Q, it_d_model) — projected query embeddings.
            memory: (B, V, P, it_d_model) — projected encoder output
                    (variables × patches).
            stage: Per-sample stage (1-4). Stages 1-2 use sinusoidal time PE,
                   stages 3-4 use rotary cycle PE. Defaults to stage 1 if None.

        Returns:
            (B, prefix_num, it_d_model) — fused prefix tokens.
        """
        B = query.size(0)

        # Prepend learnable prefix tokens to query
        x = torch.cat([self.prefix_token.expand(B, -1, -1), query], dim=1)
        x = x + self.instruc_pos(x)

        # --- Stage-aware positional encoding for memory ---
        if stage is None:
            stage_list = [1] * B
        elif torch.is_tensor(stage):
            stage_list = stage.tolist()
        else:
            stage_list = list(stage)

        cycle_idx = [i for i, s in enumerate(stage_list) if s not in (3, 4)]
        cross_cycle_idx = [i for i, s in enumerate(stage_list) if s in (3, 4)]

        # Build reorder map
        original_order = cycle_idx + cross_cycle_idx
        reorder = {idx: i for i, idx in enumerate(original_order)}
        reverse = [reorder[i] for i in range(len(stage_list))]

        parts = []
        if cycle_idx:
            sub = memory[cycle_idx]
            b, l, v, d = sub.shape
            sub = sub.view(b * l, v, d)
            sub = sub + self.time_pos(sub)
            sub = sub.view(b, l, v, d)
            parts.append(sub)

        if cross_cycle_idx:
            sub = memory[cross_cycle_idx]
            b, l, v, d = sub.shape
            sub = sub.view(b * v, l, d)
            sub = sub + self.cycle_pos(sub)
            sub = sub.view(b, l, v, d)
            parts.append(sub)

        memory = torch.cat(parts, dim=0)[reverse]

        # Variable positional encoding
        b, l, v, d = memory.shape
        memory = memory.view(b * l, v, d)
        memory = memory + self.var_pos(memory)
        memory = memory.view(b, l, v, d)

        # Decoder layers
        for layer in self.layers:
            x = layer(x, memory)

        x = self.norm(x)
        return x[:, : self.prefix_num, :]


# ---------------------------------------------------------------------------
# ITFormerModel — main model implementing BaseModel
# ---------------------------------------------------------------------------


class ITFormerModel(BaseModel):
    """Instruction-aware Time Series Transformer for time series QA.

    Uses a multi-variable encoder, an instruction-time fusion module
    with learnable prefix tokens, and token merging into a frozen LLM.
    """

    IMAGE_PAD_TOKEN = "<|image_pad|>"

    def __init__(
        self,
        device: str,
        llm_model: nn.Module,
        text_tokenizer: Any,
        ts_encoder: nn.Module,
        itformer_fusion: ITFormerFusion,
        ts_project: nn.Linear,
        query_project: nn.Linear,
        fusion_project: nn.Linear,
        freeze_ts_encoder: bool = True,
        backbone: Any = None,
    ):
        super().__init__(device)
        self.llm_model = llm_model
        # Keep the QwenBackbone instance reachable so train.py's
        # configure_training_mode can call ``model.backbone.apply_peft(...)``
        # for LoRA injection — matches the contract Flamingo + ChatTS uphold.
        self.backbone = backbone
        self.text_tokenizer = text_tokenizer
        self.ts_encoder = ts_encoder
        self.itformer_fusion = itformer_fusion
        self.ts_project = ts_project
        self.query_project = query_project
        self.fusion_project = fusion_project
        self.prefix_num = itformer_fusion.prefix_num
        self.freeze_ts_encoder = freeze_ts_encoder

        if freeze_ts_encoder:
            for p in self.ts_encoder.parameters():
                p.requires_grad = False

    @classmethod
    def from_config(cls, config: "ExperimentConfig", device: str) -> "ITFormerModel":
        # --- Backbone ---
        backbone = QwenBackbone(model_id=config.backbone.model_id)

        text_tokenizer = backbone.load_tokenizer()
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token

        # Add placeholder token (Qwen has it natively; Llama etc. do not)
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": [cls.IMAGE_PAD_TOKEN]}
        )

        llm_model = backbone.load_model(device_map={"": device})
        llm_model.resize_token_embeddings(len(text_tokenizer))
        llm_d_model = llm_model.config.hidden_size

        # --- Chronos-2 encoder (multi-variable) ---
        enc_cfg = dict(config.model.encoder)
        enc_cfg.pop("type", None)
        ts_encoder = Chronos2TSEncoder(**enc_cfg).to(device)

        # Load pretrained encoder weights if specified
        extra = config.model.extra_kwargs
        ts_encoder_ckpt = extra.get("ts_encoder_checkpoint")
        if ts_encoder_ckpt and os.path.exists(ts_encoder_ckpt):
            state = torch.load(ts_encoder_ckpt, map_location="cpu")
            # Strip 'model.' prefix if present
            cleaned = {}
            for k, v in state.items():
                cleaned[k.removeprefix("model.")] = v
            msg = ts_encoder.load_state_dict(cleaned, strict=False)
            print(f"Loaded TS encoder weights: missing={len(msg.missing_keys)}, "
                  f"unexpected={len(msg.unexpected_keys)}")

        # --- ITFormer fusion params ---
        it_d_model = extra.get("it_d_model", 896)
        it_n_heads = extra.get("it_n_heads", 16)
        it_layers = extra.get("it_layers", 2)
        it_dropout = extra.get("it_dropout", 0.1)
        prefix_num = extra.get("prefix_num", 25)
        freeze_ts_encoder = extra.get("freeze_ts_encoder", True)

        fusion = ITFormerFusion(
            it_d_model=it_d_model,
            it_n_heads=it_n_heads,
            it_layers=it_layers,
            it_dropout=it_dropout,
            prefix_num=prefix_num,
        ).to(device)

        ts_project = nn.Linear(ts_encoder.output_dim, it_d_model).to(device)
        query_project = nn.Linear(llm_d_model, it_d_model).to(device)
        fusion_project = nn.Linear(it_d_model, llm_d_model).to(device)

        # Align all trainable components to the LLM's dtype (e.g. bfloat16)
        llm_dtype = next(llm_model.parameters()).dtype
        ts_encoder.to(llm_dtype)
        fusion.to(llm_dtype)
        ts_project.to(llm_dtype)
        query_project.to(llm_dtype)
        fusion_project.to(llm_dtype)

        return cls(
            device=device,
            llm_model=llm_model,
            text_tokenizer=text_tokenizer,
            ts_encoder=ts_encoder,
            itformer_fusion=fusion,
            ts_project=ts_project,
            query_project=query_project,
            fusion_project=fusion_project,
            freeze_ts_encoder=freeze_ts_encoder,
            backbone=backbone,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        return self.text_tokenizer

    # ------------------------------------------------------------------
    # Token merging
    # ------------------------------------------------------------------

    def _merge_features(
        self,
        fused_features: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Replace ``<|image_pad|>`` positions in the input embeddings with
        fused time-series features.

        Args:
            fused_features: (B, prefix_num, llm_d_model)
            input_ids: (B, seq_len)

        Returns:
            inputs_embeds: (B, seq_len, llm_d_model)
        """
        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
        pad_id = self.text_tokenizer(self.IMAGE_PAD_TOKEN, add_special_tokens=False)["input_ids"][0]
        batch_idx, seq_idx = torch.where(input_ids == pad_id)

        B, P, D = fused_features.shape
        expected = B * P
        actual = len(batch_idx)
        if actual != expected:
            # Truncate or expand to match
            if actual > expected:
                batch_idx = batch_idx[:expected]
                seq_idx = seq_idx[:expected]
            else:
                # Pad with zeros if fewer positions than expected — shouldn't happen
                pass

        flat_features = fused_features.reshape(-1, D)[:len(batch_idx)]
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[batch_idx, seq_idx] = flat_features.to(
            dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        return inputs_embeds

    # ------------------------------------------------------------------
    # Encoding pipeline
    # ------------------------------------------------------------------

    def _encode_and_fuse(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        query_ids: torch.Tensor,
        stage: list[int] | torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the full encoding pipeline and return inputs_embeds.

        Returns:
            inputs_embeds: (B, seq_len, llm_d_model)
        """
        # Encode time series
        if self.ts_encoder.multi_variable:
            ts_out = self.ts_encoder(time_series)  # (B, V, P, d_model)
        else:
            # Single-variable encoder: process each variable independently
            B, L, V = time_series.shape
            outs = []
            for v in range(V):
                out = self.ts_encoder(time_series[:, :, v])  # (B, P, d_model)
                outs.append(out)
            ts_out = torch.stack(outs, dim=1)  # (B, V, P, d_model)
        ts_proj = self.ts_project(ts_out)  # (B, V, P, it_d_model)

        # Encode query
        query_embeds = self.llm_model.get_input_embeddings()(query_ids)
        query_proj = self.query_project(query_embeds)  # (B, Q, it_d_model)

        # Fuse
        fused = self.itformer_fusion(query_proj, ts_proj, stage)  # (B, prefix_num, it_d_model)
        fused = self.fusion_project(fused)  # (B, prefix_num, llm_d_model)

        # Merge into LLM input
        return self._merge_features(fused, input_ids)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        batch: list[dict[str, Any]],
        training: bool = True,
        normalize: bool = False,
    ) -> dict[str, torch.Tensor]:
        tokenizer = self.text_tokenizer
        pad_token = self.IMAGE_PAD_TOKEN

        # --- Stack time series: list of (L,) tensors per sample × V channels ---
        ts_list = []
        for item in batch:
            channels = item["time_series"]  # list of (L,) arrays or tensors
            tensors = [torch.as_tensor(ch, dtype=torch.float32) for ch in channels]
            ts_list.append(torch.stack(tensors, dim=-1))  # (L, V)
        # Pad to same length
        max_len = max(t.size(0) for t in ts_list)
        padded = []
        for t in ts_list:
            if t.size(0) < max_len:
                pad = torch.zeros(max_len - t.size(0), t.size(1), dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        llm_dtype = next(self.llm_model.parameters()).dtype
        time_series = torch.stack(padded).to(device=self.device, dtype=llm_dtype)  # (B, L, V)

        # --- Build prompts with <|image_pad|> placeholders ---
        text_inputs = []
        prompt_lengths = []
        query_texts = []
        stages = []

        for item in batch:
            # Build prompt text
            prompt = item["pre_prompt"]
            # Insert pad tokens for the fused features
            prompt += " " + (pad_token * self.prefix_num)
            if item["post_prompt"]:
                prompt += " " + item["post_prompt"]

            # Extract query (the question text for ITFormer's query_ids)
            question = item.get("question", item.get("post_prompt", ""))
            query_texts.append(question)

            # Extract stage (default to 1)
            stages.append(item.get("stage", 1))

            if not training:
                text_inputs.append(prompt)
                continue

            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prompt_tokens))
            full_text = prompt + " " + item["answer"]
            text_inputs.append(full_text)

        # --- Tokenize query_ids ---
        query_tokenized = tokenizer(
            query_texts, padding="longest", return_tensors="pt", add_special_tokens=False
        )
        query_ids = query_tokenized.input_ids.to(self.device)

        # --- Tokenize main text ---
        original_padding = tokenizer.padding_side
        if not training:
            tokenizer.padding_side = "left"

        tokenized = tokenizer(text_inputs, padding="longest", return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        tokenizer.padding_side = original_padding

        stage_tensor = torch.tensor(stages, dtype=torch.long, device=self.device)

        if not training:
            return {
                "time_series": time_series,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": None,
                "query_ids": query_ids,
                "stage": stage_tensor,
            }

        # --- Create labels ---
        labels = torch.full_like(input_ids, -100)
        for i, prompt_length in enumerate(prompt_lengths):
            non_pad = torch.where(input_ids[i] != tokenizer.pad_token_id)[0]
            answer_pos = non_pad[non_pad >= prompt_length]
            if len(answer_pos) > 0:
                labels[i, answer_pos] = input_ids[i, answer_pos]

        return {
            "time_series": time_series,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "query_ids": query_ids,
            "stage": stage_tensor,
        }

    def forward(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        query_ids = kwargs.get("query_ids")
        stage = kwargs.get("stage")

        if query_ids is None:
            raise ValueError("ITFormerModel.forward() requires 'query_ids' in kwargs.")

        inputs_embeds = self._encode_and_fuse(time_series, input_ids, query_ids, stage)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits

        result: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def generate(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        query_ids = generate_kwargs.pop("query_ids", None)
        stage = generate_kwargs.pop("stage", None)

        if query_ids is None:
            raise ValueError("ITFormerModel.generate() requires 'query_ids' in kwargs.")

        inputs_embeds = self._encode_and_fuse(time_series, input_ids, query_ids, stage)

        with torch.inference_mode():
            gen_ids = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.text_tokenizer.pad_token_id,
                eos_token_id=self.text_tokenizer.eos_token_id,
                **generate_kwargs,
            )

        # Strip input prefix — return answer tokens only.
        # With inputs_embeds, some transformers versions return only new
        # tokens (shape < input length); others prepend the full input.
        input_len = input_ids.shape[1]
        if gen_ids.shape[1] > input_len:
            return gen_ids[:, input_len:]
        return gen_ids

    def get_trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            "itformer_fusion": list(self.itformer_fusion.parameters()),
            "ts_project": list(self.ts_project.parameters()),
            "query_project": list(self.query_project.parameters()),
            "fusion_project": list(self.fusion_project.parameters()),
        }
        # Include encoder if not frozen
        if not self.freeze_ts_encoder:
            groups["encoder"] = list(self.ts_encoder.parameters())
        return groups

    def get_eos_token(self) -> str:
        return self.text_tokenizer.eos_token

    def get_backbone_module(self) -> nn.Module:
        return self.llm_model
