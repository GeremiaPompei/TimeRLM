"""Qwen backbone — the single LLM backbone used by all three TSLM
architectures (ChatTS, Flamingo, ITFormer). Handles tokenizer/model loading
and LoRA injection.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class QwenBackbone:
    """Backbone for Qwen 3 models (e.g., Qwen3-0.6B, Qwen3.5-4B).

    Handles tokenizer setup and model loading. Architecture-specific special
    tokens (e.g. Flamingo's ``<|endofchunk|>``) are added by the model, not here.
    """

    default_lora_targets: list[str] = ["q_proj", "v_proj"]

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", **kwargs: Any):
        self.model_id = model_id
        self.config = kwargs
        self._model = None
        self._tokenizer = None

    def load_model(self, **kwargs: Any) -> PreTrainedModel:
        """Load the Qwen model from HuggingFace.

        Args:
            **kwargs: Passed to AutoModelForCausalLM.from_pretrained().
                Common options: device_map, torch_dtype, attn_implementation.

        Returns:
            The loaded Qwen model.
        """
        defaults = {
            "local_files_only": False,
            "trust_remote_code": True,
            "cache_dir": None,
            "attn_implementation": "eager",
        }
        defaults.update(kwargs)

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **defaults)

        # Resize embeddings if tokenizer has been loaded with extra tokens
        if self._tokenizer is not None:
            self._model.resize_token_embeddings(len(self._tokenizer))

        return self._model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and configure the Qwen tokenizer.

        Qwen's tokenizer ships with a pad token (``<|endoftext|>``), so no
        special tokens need to be added. Architecture-specific special tokens
        are added by the model (e.g. Flamingo adds ``<|endofchunk|>``).

        Returns:
            Configured tokenizer.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
        )

        # Qwen already has pad_token=<|endoftext|>, but guard just in case
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            self._tokenizer.pad_token = "<PAD>"

        return self._tokenizer

    def apply_peft(self, model, peft_config: dict[str, Any]) -> None:
        """Apply LoRA adapters to the model in-place.

        Uses ``inject_adapter_in_model`` so the module tree is modified
        directly without wrapping in a ``PeftModel`` (which would break
        architectures that access LLM sub-modules by attribute).

        Args:
            model: The LLM module to inject LoRA into.
            peft_config: Dictionary with keys ``r``, ``alpha``,
                ``target_modules`` (list or None for backbone defaults),
                and ``dropout``.
        """
        from peft import LoraConfig, inject_adapter_in_model

        lora_config = LoraConfig(
            r=peft_config.get("r", 16),
            lora_alpha=peft_config.get("alpha", 32),
            target_modules=peft_config.get("target_modules") or self.default_lora_targets,
            lora_dropout=peft_config.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        inject_adapter_in_model(lora_config, model)
