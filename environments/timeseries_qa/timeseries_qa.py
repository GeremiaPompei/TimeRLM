"""Time-series QA: present a series (text or a plotted image) with a question and
score the answer 0/1. Datasets live in ``ts_datasets``; ``mode`` picks the
presentation. See README.md.
"""

from __future__ import annotations

import inspect

import verifiers as vf
from ts_datasets import DATASETS
from utils.formatter import format_timeseries, plot_timeseries

SYSTEM_PROMPT = (
    "You are an expert at analyzing time series. You are given one or "
    "more channels of time-series data followed by a question. Respond with only "
    "the answer and nothing else."
)


def _text_renderer(_system_prompt: str):
    def render(sample: dict) -> dict:
        return {"question": f"{format_timeseries(sample['series'])}\n\nQuestion: {sample['question']}"}
    return render


def _image_renderer(system_prompt: str):
    # Instruction is inlined (not a system_prompt): a string system message can't
    # share an Arrow column with list-valued image content. See mmmu.
    def render(sample: dict) -> dict:
        # When the dataset decimated the series for plot legibility, it
        # also stuffs the *original* sample count on `length` so the
        # renderer can keep the x-axis in true indices. See
        # `plot_timeseries(length=...)`.
        uris = sample.get("image_uris") or [
            plot_timeseries(sample["series"], length=sample.get("length"))
        ]
        content: list[dict] = [
            {"type": "text", "text": f"Question: {sample['question']}\n\n{system_prompt}"},
        ]
        for uri in uris:
            content.append({"type": "image_url", "image_url": {"url": uri}})
        return {"prompt": [{"role": "user", "content": content}]}
    return render


RENDERER_FACTORIES = {"text": _text_renderer, "image": _image_renderer}


def _strip_null_content_keys(messages):
    """Arrow schema unification back-fills every content part with the union of
    keys across parts (text parts gain ``image_url: None`` and vice versa).
    Lenient providers ignore the nulls; OpenAI rejects the request with
    "Unexpected keys in a message content image dict".

    By rollout time messages are Pydantic models, and the client's
    ``to_native_prompt`` dumps content parts without ``exclude_none`` — so the
    nulls must be stripped here, into plain dicts. ``model_copy`` (no
    re-validation) keeps the cleaned dicts from being coerced back into
    null-bearing part models."""
    cleaned = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            parts = []
            for p in content:
                if hasattr(p, "model_dump"):
                    p = p.model_dump(exclude_none=True)
                elif isinstance(p, dict):
                    p = {k: v for k, v in p.items() if v is not None}
                parts.append(p)
            if isinstance(msg, dict):
                msg = {**msg, "content": parts}
            else:
                msg = msg.model_copy(update={"content": parts})
        cleaned.append(msg)
    return cleaned


class _MultimodalSingleTurnEnv(vf.SingleTurnEnv):
    """SingleTurnEnv that sanitizes Arrow-unified multimodal prompts."""

    async def get_model_response(self, state, prompt, **kwargs):
        if isinstance(prompt, list):
            prompt = _strip_null_content_keys(prompt)
        return await super().get_model_response(state, prompt, **kwargs)


def _dataset_metric_funcs(ds, parser):
    """Wrap each metric the dataset exposes via ``extra_metric_names`` into a
    SingleTurnEnv-shaped rubric func ``(parser, completion, answer) -> float``.

    The dataset owns metric definitions; the env only adapts the calling
    convention.
    """
    names = tuple(getattr(ds, "extra_metric_names", ()))

    def _make(name):
        def fn(parser, completion, answer) -> float:
            parsed = parser.parse_answer(completion) or ""
            return float(ds.score_metrics(parsed, answer).get(name, 0.0))
        fn.__name__ = name
        return fn

    return [_make(n) for n in names]


def load_environment(
    dataset: str = "anomalyxl",
    split: str | None = None,
    eval_split: str | None = None,
    mode: str = "text",
    use_image: bool = True,
    enable_sub_llms: bool = False,
    sub_llm_max_turns: int = 5,
    sub_model: str | None = None,
    image_via_user_followup: bool = False,
    use_local_executor: bool = False,
    max_turns: int | None = None,
    seed_images: bool = False,
    **dataset_kwargs,
) -> vf.Environment:
    """``dataset``: key in DATASETS. ``mode``: "text", "image", or "rlm". ``split``/
    ``eval_split`` default per dataset. ``use_image`` only applies to ``rlm`` mode:
    True registers the ``look`` image tool, False is a pure-numeric REPL.
    ``enable_sub_llms=True`` (RLM mode only) gives the root model an ``llm_batch()``
    tool to spawn depth-1 sub-LLM calls; ``sub_llm_max_turns`` caps each child's
    tool-call loop; ``sub_model`` overrides which model serves the children.
    Extra kwargs go to the dataset class (e.g. ``eval_size``, ``seed``, ``interval``)."""
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset!r}. Available: {', '.join(sorted(DATASETS))}.")
    valid_modes = set(RENDERER_FACTORIES) | {"rlm"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown mode {mode!r}. Available: {', '.join(sorted(valid_modes))}.")

    cls = DATASETS[dataset]
    split = split or cls.default_split
    eval_split = eval_split if eval_split is not None else cls.default_eval_split
    for s in filter(None, (split, eval_split)):
        if s not in cls.splits:
            raise ValueError(f"Split {s!r} not available for {dataset!r}; choose from {cls.splits}.")

    # Forward ``mode`` only to datasets that *explicitly* declare it (e.g.
    # ARFBench tunes series-loading and system prompt per mode). Inspect the
    # signature rather than catching TypeError: a dataset whose ``__init__``
    # takes ``**load_kwargs`` would silently absorb ``mode`` and leak it into
    # ``load_dataset()`` (which datasets>=3 rejects).
    if "mode" in inspect.signature(cls).parameters:
        ds = cls(mode=mode, **dataset_kwargs)
    else:
        ds = cls(**dataset_kwargs)

    if mode == "rlm":  # lazy import
        from rlm import make_rlm_env

        return make_rlm_env(
            ds,
            split,
            eval_split,
            use_image=use_image,
            enable_sub_llms=enable_sub_llms,
            sub_llm_max_turns=sub_llm_max_turns,
            sub_model=sub_model,
            image_via_user_followup=image_via_user_followup,
            use_local_executor=use_local_executor,
            max_turns=max_turns,
            seed_images=seed_images,
        )

    prompt = ds.system_prompt or SYSTEM_PROMPT
    render = RENDERER_FACTORIES[mode](prompt)
    parser = vf.Parser()

    def correct_answer(parser, completion, answer) -> float:
        parsed_answer = parser.parse_answer(completion) or ""
        try:
            return ds.score_answer(parsed_answer, answer)
        except Exception:
            # A malformed answer must score zero, never take down the eval. The
            # scorers already reject non-finite fields; this is the backstop for
            # anything they don't anticipate.
            return 0.0

    metric_funcs = _dataset_metric_funcs(ds, parser)
    rubric_funcs = [correct_answer, *metric_funcs]
    rubric_weights = [1.0] + [0.0] * len(metric_funcs)
    rubric = vf.Rubric(funcs=rubric_funcs, weights=rubric_weights, parser=parser)

    # Lazy builders: materialize the dataset on first use, not at load time.
    env_kwargs = dict(dataset=lambda: ds.build(split, render), parser=parser, rubric=rubric)
    if mode == "text":  # image mode inlines the instruction in the user content
        env_kwargs["system_prompt"] = prompt
    if eval_split is not None:
        env_kwargs["eval_dataset"] = lambda: ds.build(eval_split, render)

    env_cls = _MultimodalSingleTurnEnv if mode == "image" else vf.SingleTurnEnv
    return env_cls(**env_kwargs)
