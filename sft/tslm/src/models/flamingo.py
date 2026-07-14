"""OpenTSLMFlamingo — gated cross-attention time series LLM implementing BaseModel.

Data flow:
    time_series (B,channels,L)  →  Chronos2TSEncoder  →  (B,N,d_model)
        →  PerceiverResampler (64 latents)  →  (B,T,M,d_model)
            →  gated cross-attention, interleaved every N decoder layers
                →  LLM forward  →  loss
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING

import torch
import torch._dynamo
import torch.nn as nn

from open_flamingo import Flamingo
from open_flamingo.src.flamingo_lm import FlamingoLMMixin, FlamingoLayer
from open_flamingo.src.helpers import PerceiverResampler
from open_flamingo.src.utils import extend_instance
from einops import rearrange

from src.models.base import BaseModel
from src.models.chronos2_encoder import Chronos2TSEncoder
from src.models.qwen_backbone import QwenBackbone
from src.prompt.full_prompt import FullPrompt
from src.datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig


# Decoder-layer attribute lookup, used to auto-detect where a HF causal-LM
# keeps its transformer block list (needed by open_flamingo's FlamingoLMMixin
# to interleave gated cross-attention layers).
_KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
    "gemma": "model.layers",
    "gemma2": "model.layers",
    "gemma3": "model.layers",
    "medgemma": "model.layers",
    "qwen": "model.layers",
}


def _infer_decoder_layers_attr_name(model: nn.Module) -> str:
    model_class_name = model.__class__.__name__
    if "gemma3" in model_class_name.lower():
        if "ConditionalGeneration" in model_class_name:
            return "language_model.layers"
        return "model.layers"

    for key in _KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if key.lower() in model_class_name.lower():
            return _KNOWN_DECODER_LAYERS_ATTR_NAMES[key]

    raise ValueError(
        f"Cannot infer decoder layers attribute for {model_class_name}. "
        "Please add it to _KNOWN_DECODER_LAYERS_ATTR_NAMES."
    )


# Monkey-patch FlamingoLayer to add attention_type / layer_type properties
# for compatibility with newer transformers. Qwen3-family backbones use
# hybrid attention dispatchers that read ``layer.layer_type`` per decoder
# layer; without this proxy every forward pass throws AttributeError.
def _attention_type_property(self):
    """Proxy the attention_type attribute from the underlying decoder layer."""
    return getattr(self.decoder_layer, "attention_type", None)


def _layer_type_property(self):
    """Proxy the layer_type attribute from the underlying decoder layer."""
    return getattr(self.decoder_layer, "layer_type", None)


FlamingoLayer.attention_type = property(_attention_type_property)  # type: ignore
FlamingoLayer.layer_type = property(_layer_type_property)  # type: ignore


class TimeSeriesFlamingoWithTrainableEncoder(Flamingo):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        num_latents: int = 64,
        projector_instance: nn.Module | None = None,
    ):
        """
        TimeSeriesFlamingoWithTrainableEncoder with configurable projector.

        Args:
            num_latents: Number of latent tokens (M) in the Perceiver Resampler.
                Controls the compression ratio of time series features.
                Default is 64 for backwards compatibility.
            projector_instance: A pre-built projector module. When provided,
                replaces the default PerceiverResampler.
        """
        super().__init__(vision_encoder, lang_encoder, eoc_token_id, media_token_id, vis_dim, cross_attn_every_n_layers, gradient_checkpointing)

        if projector_instance is not None:
            self.perceiver = projector_instance
        elif num_latents != 64:
            # Override perceiver with configurable num_latents for ablation studies
            # The base Flamingo class hardcodes num_latents=64, so we replace it here
            self.perceiver = PerceiverResampler(dim=vis_dim, num_latents=num_latents)

        # PerceiverResampler expects 5D input (b, T, F, p, d) and reduces p → M.
        # Linear/MLP projectors work on the last dim only and expect 4D (b, T, tokens, d).
        self._projector_handles_5d = isinstance(self.perceiver, PerceiverResampler)

    # Override the _encode_vision_x method to handle time series data
    # In the original Flamingo, the vision_encoder is a CLIPModel, which is not trainable (with torch.no_grad())
    # Here, we use a time-series encoder, which is trainable
    def _encode_vision_x(self, vision_x):
        # Handle time series data while still using the trainable TS encoder
        if vision_x.ndim == 4:  # For shape (b, T_img, F, features)
            b, T, F, features = vision_x.shape

            if getattr(self.vision_encoder, "multi_variable", False):
                # Multi-variable encoder: pass all channels together
                vision_x = rearrange(vision_x, "b T F c -> (b T) c F")  # (b*T, L, V)
                vision_x = self.vision_encoder(vision_x)  # (b*T, V, N, D)
                vision_x = rearrange(vision_x, "(b T) V N D -> b T V N D", b=b, T=T)
                # Flatten V and N for the projector: treat like (b, T, V*N, D)
                V, N = vision_x.shape[2], vision_x.shape[3]
                vision_x = rearrange(vision_x, "b T V N D -> b T (V N) D")
                # Expand F dim for compatibility: (b, T, 1, V*N, D)
                vision_x = vision_x.unsqueeze(2)
                F = 1  # override F for downstream reshape
            else:
                # Single-channel encoder: process channels independently (existing path)
                vision_x = rearrange(vision_x, "b T F c -> (b T F) c")
                vision_x = self.vision_encoder(vision_x)  # [(b*T*F), patches, features]
                vision_x = rearrange(vision_x, "(b T F) p d -> b T F p d", b=b, T=T, F=F)

            # Process through projector (perceiver or linear/mlp)
            if self._projector_handles_5d:
                vision_x = self.perceiver(vision_x)
            else:
                vision_x = rearrange(vision_x, "b T F p d -> b T (F p) d")
                vision_x = self.perceiver(vision_x)

        else:
            # Original image processing path
            assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
            b, T, F = vision_x.shape[:3]
            assert F == 1, "Only single frame supported"

            vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")

            vision_x = self.vision_encoder(vision_x)[1]
            vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

            if self._projector_handles_5d:
                vision_x = self.perceiver(vision_x)
            else:
                vision_x = rearrange(vision_x, "b T F p d -> b T (F p) d")
                vision_x = self.perceiver(vision_x)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)


class OpenTSLMFlamingo(BaseModel):
    # Flamingo-specific special tokens
    EOC_TOKEN = "<|endofchunk|>"    # End-of-chunk / answer terminator
    MEDIA_TOKEN = "<image>"          # Time series placeholder

    def __init__(
        self,
        device: str,
        backbone: QwenBackbone,
        cross_attn_every_n_layers: int = 1,
        freeze_lm_embeddings: bool = False,
        encoder_kwargs: dict | None = None,
        projector_kwargs: dict | None = None,
        **flamingo_kwargs,
    ):
        """
        Initialize OpenTSLM Flamingo model.

        Args:
            device: Device to run the model on ('cuda', 'cpu', 'mps')
            backbone: A QwenBackbone instance that handles tokenizer/model loading.
            cross_attn_every_n_layers: Insert cross-attention every N layers
            freeze_lm_embeddings: Whether to freeze LM input embeddings
            encoder_kwargs: Keyword arguments forwarded to Chronos2TSEncoder
                (model_id, freeze, ...).
            projector_kwargs: Keyword arguments forwarded to PerceiverResampler
                (num_latents, ...).
            **flamingo_kwargs: Additional arguments passed to the Flamingo model
        """
        super().__init__(device)
        self.freeze_lm_embeddings = freeze_lm_embeddings
        print(f"Flamingo Using device: {self.device}")

        self.backbone = backbone

        time_series_encoder = Chronos2TSEncoder(**(encoder_kwargs or {})).to(device)

        # Delegate tokenizer and model loading to the backbone
        text_tokenizer = backbone.load_tokenizer()

        # Add Flamingo-specific special tokens
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.EOC_TOKEN, self.MEDIA_TOKEN]}
        )

        lang_encoder = backbone.load_model(device_map={"": device})

        # convert LM to FlamingoLM
        extend_instance(lang_encoder, FlamingoLMMixin)

        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        lang_encoder.resize_token_embeddings(len(text_tokenizer))

        # Fix compatibility for Gemma3Config which has hidden_size in text_config
        if hasattr(lang_encoder.config, "text_config") and hasattr(
            lang_encoder.config.text_config, "hidden_size"
        ):
            if not hasattr(lang_encoder.config, "hidden_size"):
                lang_encoder.config.hidden_size = (
                    lang_encoder.config.text_config.hidden_size
                )

        # --- Perceiver Resampler projector ---
        projector_instance = PerceiverResampler(
            dim=time_series_encoder.output_dim, **(projector_kwargs or {})
        )
        # Dimension-preserving, so output dim == input dim.
        projector_output_dim = getattr(
            projector_instance, "output_dim", time_series_encoder.output_dim
        )

        model = TimeSeriesFlamingoWithTrainableEncoder(
            SimpleNamespace(visual=time_series_encoder),
            lang_encoder,
            text_tokenizer.encode(self.EOC_TOKEN)[-1],
            text_tokenizer.encode(self.MEDIA_TOKEN)[-1],
            vis_dim=projector_output_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            projector_instance=projector_instance,
            **flamingo_kwargs,
        )

        # Ensure vision_encoder is a PyTorch module (not a SimpleNamespace)
        if not hasattr(model.vision_encoder, 'parameters'):
            if hasattr(model.vision_encoder, 'visual'):
                model.vision_encoder = model.vision_encoder.visual
            else:
                raise ValueError(
                    f"vision_encoder is not a PyTorch module. Got type: {type(model.vision_encoder)}"
                )

        # Align all components to the LLM's dtype (e.g. bf16). The lang_encoder
        # loads in its native dtype but vision_encoder, perceiver, and
        # cross-attn layers initialize as float32.
        llm_dtype = next(lang_encoder.parameters()).dtype
        model.vision_encoder.to(llm_dtype)
        model.perceiver.to(llm_dtype)
        model.lang_encoder.gated_cross_attn_layers.to(llm_dtype)

        self.model = model
        self.llm = model
        self.text_tokenizer = text_tokenizer

    @classmethod
    def from_config(cls, config: "ExperimentConfig", device: str) -> "OpenTSLMFlamingo":
        """Construct a Flamingo model from an ExperimentConfig."""
        enc_cfg = dict(config.model.encoder)
        enc_cfg.pop("type", None)
        flam = config.model.extra_kwargs
        proj_cfg = dict(config.model.projector)
        proj_cfg.pop("type", None)

        # Backward compat: merge num_latents from extra_kwargs if not in projector config
        if "num_latents" not in proj_cfg:
            num_latents_compat = flam.get("num_latents")
            if num_latents_compat is not None:
                proj_cfg["num_latents"] = num_latents_compat

        backbone = QwenBackbone(model_id=config.backbone.model_id)

        return cls(
            device=device,
            backbone=backbone,
            encoder_kwargs=enc_cfg,
            projector_kwargs=proj_cfg,
            cross_attn_every_n_layers=flam.get("cross_attn_every_n_layers", 1),
            freeze_lm_embeddings=flam.get("freeze_lm_embeddings", False),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        return self.text_tokenizer

    # ------------------------------------------------------------------
    # BaseModel abstract method implementations
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        batch: list[dict[str, Any]],
        training: bool = True,
        normalize: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Convert a list of dataset sample dicts into tensors for Flamingo.

        Handles patch-alignment, time-series padding, prompt tokenization
        with Flamingo's ``<image>`` / ``<|endofchunk|>`` special tokens,
        and label creation.

        Args:
            batch: List of sample dicts from the dataset.
            training: If True, include the answer text and create labels.
                If False, tokenize prompt only (for generation).
            normalize: If True, normalize each time series to zero mean and
                unit variance before patch alignment.

        Returns:
            Dict with keys ``time_series``, ``input_ids``,
            ``attention_mask``, and ``labels`` (None when not training).
        """
        # Convert raw time-series lists → stacked tensors and optionally
        # pad to a multiple of patch_size. Chronos-2 handles its own
        # patching, so we use patch_size=1 as a no-op alignment that still
        # does the list→tensor conversion needed downstream.
        patch_size = getattr(self.model.vision_encoder, "patch_size", 1)
        batch = extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=patch_size, normalize=normalize,
        )

        def pad_time_series(batch, max_length=None):
            """Pad time series to the same length (either max in batch or specified max)"""
            time_series = [item["time_series"] for item in batch]

            # Determine target length (either specified or max in batch)
            if max_length is None:
                max_length = max(ts.shape[1] for ts in time_series)

            padded_series = []
            for ts in time_series:
                current_length = ts.shape[1]
                if current_length < max_length:
                    padding_shape = list(ts.shape)
                    padding_shape[1] = max_length - current_length
                    padding = torch.zeros(
                        padding_shape, device=ts.device, dtype=ts.dtype
                    )
                    padded = torch.cat([ts, padding], dim=1)
                else:
                    padded = ts[:, :max_length]

                padded_series.append(padded)

            return torch.stack(padded_series)

        cast_dtype = next(self.model.lang_encoder.parameters()).dtype
        tokenizer = self.text_tokenizer
        media_token_id = tokenizer(self.MEDIA_TOKEN, add_special_tokens=False)["input_ids"][-1]
        endofchunk_token_id = tokenizer(self.EOC_TOKEN, add_special_tokens=False)[
            "input_ids"
        ][-1]

        # Process time series data — shape (B, channels, seq_len)
        time_series = pad_time_series(batch).to(
            self.device, dtype=cast_dtype, non_blocking=True
        )

        # Process text inputs
        text_inputs = []
        prompt_lengths = []

        for item in batch:
            # Build the prompt text without answer
            prompt_text = item["pre_prompt"]
            for ts_text in item["time_series_text"]:
                prompt_text += f" {tokenizer.decode([media_token_id])} {ts_text} {tokenizer.decode([endofchunk_token_id])}"
            if item["post_prompt"]:
                prompt_text += f" {item['post_prompt']}"

            if not training:
                # Eval/generation: prompt only, no labels
                text_inputs.append(prompt_text)
                continue

            # Training: store prompt length, then append answer
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prompt_tokens))

            full_text = prompt_text + f" {item['answer']}"
            text_inputs.append(full_text)

        # Tokenize — use left-padding for generation so the last real token
        # is always at the rightmost position (required by decoder-only LLMs).
        # Training keeps right-padding for correct label index alignment.
        original_padding_side = tokenizer.padding_side
        if not training:
            tokenizer.padding_side = "left"

        tokenized = tokenizer(text_inputs, padding="longest", return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device, non_blocking=True)
        attention_mask = tokenized.attention_mask.to(self.device, non_blocking=True)

        tokenizer.padding_side = original_padding_side

        if not training:
            return {
                "time_series": time_series,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": None,
            }

        # Create labels matrix (-100 for masked tokens)
        labels = torch.full_like(input_ids, -100)

        for i, prompt_length in enumerate(prompt_lengths):
            non_padding_indices = torch.where(input_ids[i] != tokenizer.pad_token_id)[0]
            answer_indices = non_padding_indices[non_padding_indices >= prompt_length]

            if len(answer_indices) > 0:
                labels[i, answer_indices] = input_ids[i, answer_indices]

        return {
            "time_series": time_series,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def forward(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Flamingo forward pass.

        Args:
            time_series: (B, channels, seq_len)
            input_ids: (B, text_len)
            attention_mask: (B, text_len)
            labels: (B, text_len) with -100 for ignored positions.

        Returns:
            Dict with ``'loss'`` (scalar tensor) when labels are provided.
        """
        # Flamingo expects (B, T, channels, seq_len) with T=1 time dimension
        vision_x = time_series.unsqueeze(1)

        output = self.model(
            vision_x=vision_x,
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": output[0]}

    def generate(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Generate token IDs given time series and prompt tensors.

        Args:
            time_series: (B, channels, seq_len)
            input_ids: (B, prompt_len)
            attention_mask: (B, prompt_len)
            **generate_kwargs: Forwarded to the LLM's generate method
                (e.g. max_new_tokens, temperature).

        Returns:
            Generated token IDs of shape (B, generated_len), with
            input prompt tokens stripped.
        """
        # Temporarily disable compilation to avoid data-dependent operation issues
        original_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True

        try:
            with torch.inference_mode():
                vision_x = time_series.unsqueeze(1)

                # Note: pad_token_id is not passed — open-flamingo's
                # Flamingo.generate() has a fixed parameter list and
                # does not accept it as a kwarg.
                gen_ids = self.llm.generate(
                    vision_x=vision_x,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    **generate_kwargs,
                )

                # Strip input prompt tokens — return answer only
                answer_only_ids = gen_ids[:, input_ids.shape[1]:]
                return answer_only_ids
        finally:
            torch._dynamo.config.disable = original_disable

    def get_backbone_module(self) -> nn.Module:
        """Return the LLM backbone for framework-managed LoRA."""
        return self.model.lang_encoder

    def get_trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        """Return Flamingo's trainable parameter groups.

        Respects encoder-level freezing: if the encoder has ``freeze=True``
        (e.g. pretrained Chronos-2 with ``freeze: true`` in config), its
        parameters are excluded so the training script won't re-enable them.
        """
        groups: dict[str, list[nn.Parameter]] = {}
        if not getattr(self.model.vision_encoder, "freeze", False):
            groups["encoder"] = list(self.model.vision_encoder.parameters())
        groups["projector"] = list(self.model.perceiver.parameters())
        groups["cross_attention"] = list(
            self.model.lang_encoder.gated_cross_attn_layers.parameters()
        )
        if not self.freeze_lm_embeddings:
            groups["embeddings"] = list(
                self.model.lang_encoder.get_input_embeddings().parameters()
            )
        return groups

    def get_eos_token(self) -> str:
        return self.EOC_TOKEN

    # ------------------------------------------------------------------
    # Flamingo-specific methods (not part of BaseModel interface)
    # ------------------------------------------------------------------

    def store_to_file(self, path: str = "best_model.pt"):
        state_dict = {
            "model_state": self.state_dict(),
        }
        torch.save(state_dict, path)
        print(f"Model saved to {path}")

    def load_from_file(self, path: str = "best_model.pt"):
        """
        Load model parameters with non-strict loading to handle Flamingo-specific layers.

        Supports three checkpoint formats:
          - ``{"model_state": self.state_dict()}`` — current format (keys prefixed
            with ``model.``).  Loaded via ``self.load_state_dict()``.
          - ``{"llm": self.llm.state_dict()}`` — legacy format (un-prefixed keys).
            Loaded into ``self.model`` directly.
          - Training-script checkpoints that store ``model.state_dict()`` under
            ``"model_state"`` are identical to the current format.
        """
        checkpoint = torch.load(path, map_location=self.device)

        if "llm" in checkpoint:
            # Legacy format: un-prefixed keys -> load into self.model
            model_state = checkpoint["llm"]
            load_target = self.model
        elif "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
            # Detect whether keys are prefixed (from self.state_dict()) or
            # un-prefixed (from self.model.state_dict()).
            sample_key = next(iter(model_state), "")
            if sample_key.startswith("model.") or sample_key.startswith("llm."):
                # Prefixed keys -> load via self.load_state_dict after
                # normalising any "llm." prefix to "model."
                normalised = {}
                for k, v in model_state.items():
                    if k.startswith("llm."):
                        k = "model." + k[4:]
                    normalised[k] = v
                model_state = normalised
                load_target = self
            else:
                # Un-prefixed keys -> load into self.model
                load_target = self.model
        else:
            raise RuntimeError("No recognized model state key in checkpoint.")

        # Handle pos_embed interpolation for different max_patches sizes
        # The key may or may not have the "model." prefix depending on load_target
        for pos_embed_key in ("vision_encoder.pos_embed", "model.vision_encoder.pos_embed"):
            if pos_embed_key in model_state:
                checkpoint_pos_embed = model_state[pos_embed_key]
                model_pos_embed = self.model.vision_encoder.pos_embed

                if checkpoint_pos_embed.shape != model_pos_embed.shape:
                    print(f"Interpolating pos_embed: {checkpoint_pos_embed.shape} -> {model_pos_embed.shape}")
                    pos_embed_t = checkpoint_pos_embed.transpose(1, 2)
                    target_patches = model_pos_embed.shape[1]
                    pos_interp = torch.nn.functional.interpolate(
                        pos_embed_t, size=target_patches, mode='linear', align_corners=True
                    )
                    model_state[pos_embed_key] = pos_interp.transpose(1, 2)
                break

        # Load state dict with strict=False to handle missing/unexpected keys
        missing_keys, unexpected_keys = load_target.load_state_dict(model_state, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys when loading checkpoint:")
            for key in missing_keys[:10]:
                print(f"   - {key}")
            if len(missing_keys) > 10:
                print(f"   ... and {len(missing_keys) - 10} more keys")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading checkpoint:")
            for key in unexpected_keys[:10]:
                print(f"   - {key}")
            if len(unexpected_keys) > 10:
                print(f"   ... and {len(unexpected_keys) - 10} more keys")
        self.to(self.device)

    def eval_prompt(
        self, prompt: FullPrompt, max_new_tokens: int = 1000, normalize: bool = False
    ) -> str:
        """
        Evaluate a prompt and return the generated text.
        """
        # Temporarily disable compilation to avoid data-dependent operation issues
        original_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        try:
            batch = [prompt.to_dict()]
            self.eval()
            eval_inputs = self.prepare_batch(batch, training=False, normalize=normalize)
            print("Generating")
            token_ids = self.generate(
                eval_inputs["time_series"],
                eval_inputs["input_ids"],
                eval_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
            output = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            print(f"Generated output: {output[0]}")
            return output[0]
        finally:
            # Restore original compilation setting
            torch._dynamo.config.disable = original_disable
