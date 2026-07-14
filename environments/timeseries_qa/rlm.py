"""RLM mode: sandboxed Python REPL (numpy + scipy, plus matplotlib when images
are enabled) with the series in a file.

With ``use_image=True`` the ``look(path)`` tool injects an image directly into the
model's context — no sub-LLM call. The model saves a plot inside a REPL call, then
calls ``look`` as a separate tool call; the image appears inline in its next turn
so it can continue reasoning with full visual context. With ``use_image=False`` the
``look`` tool is not registered and the REPL is purely numeric.

    # In call_python_repl:
    plt.plot(vals); plt.savefig("p.png")

    # As a direct tool call (NOT inside REPL code):
    look(path="p.png")

    # Image appears in model context → model continues in call_python_repl:
    answer['content'] = '(b)'; answer['ready'] = True

Uses the experimental RLMEnv; code runs in a Prime sandbox (not offline).
"""

from __future__ import annotations

import ast
import json
import re
import shlex

import verifiers as vf
from ts_datasets import TimeSeriesQADataset
try:
    from verifiers.envs.experimental.rlm_env import RLMEnv
except ModuleNotFoundError:
    # prime-rl's newer verifiers removed the legacy module (PR #1515); fall back
    # to our vendored copy so RL can run in that venv. See rlm_env_vendored.py.
    from rlm_env_vendored import RLMEnv
from verifiers.types import Messages, State

_SAFE_PATH = re.compile(r"[\w./-]+")

# Match `content="..."` / `content: "..."` / `content='...'` etc. — used as a
# regex fallback when the model writes its final answer as plain text instead
# of executing it in the REPL.
_CONTENT_RE = re.compile(
    r"""['"]?content['"]?\s*[:=]\s*['"]([^'"]+)['"]"""
)


def _final_answer_from(state: State, completion: Messages) -> str:
    """Best-effort recovery of the model's final answer.

    Preferred path: the model executes ``answer['content'] = '...'; answer['ready']
    = True`` inside a ``call_python_repl`` call, which populates ``state['final_answer']``.

    Fallbacks, in order: scan the tail assistant message for a JSON dict, a
    Python dict literal, a ``content="..."`` regex hit, then finally return the
    raw text. The dataset's ``score_answer`` is dispatch-on-gold-shape and
    tolerant — it will pluck the right token out of "Yes, the answer is no." or
    "Final answer: 00:00:00.875" without help. The earlier-stage structured
    parsers only exist so we don't waste a regex scan when the model emits a
    clean dict.

    Important: when the model misuses the answer machinery — e.g. reassigns
    ``answer`` to a fresh dict instead of mutating ``answer['content']`` — the
    sandbox writes a JSON-incompatible payload back, ``read_answer`` returns
    something like ``'{"ready": false, "content": ""}'`` or a Python repr, and
    ``state['final_answer']`` ends up non-empty but useless. In those cases the
    real answer is still in the *last assistant message*. To survive that, we
    consider both candidates and prefer whichever contains a JSON-shaped blob.
    """
    fa_state = _normalize_answer_value(state.get("final_answer")) if isinstance(state, dict) else ""
    fa_completion = _extract_completion_final(completion)

    # Prefer state's value when it contains a JSON object — the model
    # successfully wrote answer['content']. Otherwise prefer the completion's
    # last assistant text if it contains a JSON object; this rescues rollouts
    # where the model emitted its JSON answer as plain text but failed to
    # persist it through the sandbox's answer dict.
    if _has_json_object(fa_state):
        return fa_state
    if _has_json_object(fa_completion):
        return fa_completion
    # Last-ditch fallback: scan REPL tool outputs. The model often reassigns
    # ``answer = {<schema fields>, "ready": True}`` instead of mutating
    # ``answer['content']``; the sandbox then writes a payload without a
    # ``content`` key, ``read_answer`` returns ``""``, and the actual answer
    # only survives as the REPL's repr in the next tool message
    # (``Out[N]: {'magnitude_sigma': 49.55, 'ready': True}``). Catching that
    # here lets the strict scorer score the model's genuine prediction.
    fa_tool = _extract_completion_final_from_tools(completion)
    if _has_json_object(fa_tool):
        return fa_tool
    return fa_state or fa_completion or fa_tool


def _normalize_answer_value(value: object) -> str:
    """Coerce sandbox answer payloads to the plain text expected by scorers.

    RLM models often set ``answer['content']`` to a Python dict instead of a
    JSON string, or reassign ``answer = {'anomalies': ..., 'ready': True}``.
    The downstream dataset scorers intentionally consume text, so normalize
    these wrappers here rather than making every metric function type-check.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(stripped)
            except (ValueError, SyntaxError):
                continue
            if isinstance(obj, (dict, list)):
                normalized = _normalize_answer_value(obj)
                if normalized:
                    return normalized
        return stripped
    if isinstance(value, dict):
        content = value.get("content")
        if content not in (None, ""):
            return _normalize_answer_value(content)
        if set(value.keys()) <= {"ready", "content"}:
            return ""
        payload = dict(value)
        payload.pop("ready", None)
        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):
            return str(payload)
    if isinstance(value, list):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value).strip()


def _has_json_object(text: str) -> bool:
    """Return True if ``text`` contains a JSON object that's plausibly a real
    answer — i.e., parses, is non-empty, and isn't the sandbox-init
    ``{"ready": ..., "content": ""}`` shell. Used to decide whether to prefer
    ``state['final_answer']`` over the completion's last assistant message.
    """
    if not text or "{" not in text or "}" not in text:
        return False
    start = text.find("{")
    end = text.rfind("}") + 1
    try:
        obj = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return False
    if not isinstance(obj, dict) or not obj:
        return False
    # Sandbox-init shape `{"ready": <bool>, "content": ""}` — when the model
    # never set ``answer['content']`` it ends up here, and treating it as the
    # final answer would mask the model's actual message.
    if set(obj.keys()) <= {"ready", "content"} and not obj.get("content"):
        return False
    return True


def _msg_field(msg: object, key: str, default=None):
    """Read a message field from either a plain dict (loaded JSONL) or a
    Pydantic ``CustomBaseModel`` (live runtime). The base ``CustomBaseModel``
    defines ``.get()`` that maps to ``getattr``, but it isn't a ``dict``
    subclass — ``isinstance(msg, dict)`` rejects live messages and the
    completion scan silently finds no assistant content. Bypass that by
    duck-typing on ``.get``.
    """
    if isinstance(msg, dict):
        return msg.get(key, default)
    getter = getattr(msg, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(msg, key, default)


_OUT_REPR_RE = re.compile(r"Out\[\d+\]:\s*(.+)", re.DOTALL)


def _extract_completion_final_from_tools(completion: Messages) -> str:
    """Walk tool messages from the end and recover a dict-shaped answer that
    survives only as a REPL repr. Handles the ``Out[N]: {'key': value, ...}``
    Python-repr form by ``ast.literal_eval`` + re-serializing as JSON so the
    strict downstream parser succeeds.

    Skips empty tool outputs, ``(no output)`` markers, and image-content lists.
    """
    if not completion:
        return ""
    for msg in reversed(list(completion)):
        if _msg_field(msg, "role") != "tool":
            continue
        text = _msg_field(msg, "content")
        if not isinstance(text, str):
            continue
        m = _OUT_REPR_RE.search(text)
        if not m:
            continue
        candidate = m.group(1).strip()
        try:
            obj = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(obj, dict) or not obj:
            continue
        normalized = _normalize_answer_value(obj)
        if normalized:
            return normalized
    return ""


def _extract_completion_final(completion: Messages) -> str:
    """Scan the tail assistant message and apply the historical content-extraction
    fallbacks (`{"content": "..."}` dicts, ``content="..."`` regex). Returns the
    stripped text when no structured form is found, ``""`` when there's no
    assistant message to scan.
    """
    if not completion:
        return ""
    for msg in reversed(list(completion)):
        if _msg_field(msg, "role") != "assistant":
            continue
        raw_text = _msg_field(msg, "content")
        text = _normalize_answer_value(raw_text)
        if not text:
            continue
        stripped = text.strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(stripped)
            except (ValueError, SyntaxError):
                continue
            if isinstance(obj, (dict, list)):
                normalized = _normalize_answer_value(obj)
                if normalized:
                    return normalized
        m = _CONTENT_RE.search(stripped)
        if m:
            return _normalize_answer_value(m.group(1))
        return stripped
    return ""


def _dataset_metric_funcs_rlm(ds):
    """Wrap each metric the dataset exposes via ``extra_metric_names`` into an
    RLM-shaped rubric func ``(completion, answer, state) -> float``.

    The dataset owns metric definitions; the env only adapts the calling
    convention (RLM rubrics receive ``state`` and read the final answer from
    it via ``_final_answer_from``).
    """
    names = tuple(getattr(ds, "extra_metric_names", ()))

    def _make(name):
        def fn(completion, answer, state) -> float:
            text = _final_answer_from(state, completion)
            return float(ds.score_metrics(text, answer).get(name, 0.0))
        fn.__name__ = name
        return fn

    return [_make(n) for n in names]


def _data_note(use_image: bool, values_note: str, answer_format: str, sub_llm_nudge: str = "",
               preamble: str = "") -> str:
    """Scaffolding note prepended to each question.

    ``values_note``: dataset-specific sentence about value shape (e.g. NaN gaps).
    ``answer_format``: how the model must submit the final answer (varies because
    different datasets have different gold formats — see TimeSeriesQADataset).
    ``sub_llm_nudge``: dataset-specific guidance on when to use ``llm_batch``;
    caller supplies "" when sub-LLMs are disabled or the dataset has nothing
    extra to say.
    ``preamble``: dataset-owned lead-in sentence (see
    ``TimeSeriesQADataset.rlm_task_preamble``).
    """
    base = (
        preamble
        + "The time series is in the file `context.json` in your working directory, shaped "
        'as {"series": {channel: [values], ...}}. Load it with json. '
        + values_note
        + "Always read the actual channel keys from the data (do not guess names). "
    )
    if use_image:
        analyze = (
            "Analyze it with numpy, scipy, and matplotlib. To inspect a plot visually: save "
            "it inside a call_python_repl call (plt.savefig('p.png')), then call the `look` "
            "tool directly as a separate tool call — the image will appear inline in your "
            "context so you can continue reasoning. "
        )
    else:
        analyze = "Analyze it with numpy and scipy. "
    submit = (
        "When you have the answer, you MUST execute it in a call_python_repl tool call "
        "(writing it as text has no effect): " + answer_format
    )
    return base + analyze + sub_llm_nudge + submit


def _make_render(note: str, seed_images: bool = False):
    def _render(sample: dict) -> dict:
        out = {"info": json.dumps({"context": {"series": sample["series"]}})}
        uris = sample.get("image_uris") if seed_images else None
        if uris:
            # Hybrid config: the benchmark's official plot(s) arrive in the
            # first user turn (like single-pass image mode), and the REPL is
            # available for numeric follow-up. Scaffolding is injected into
            # this message by the env. Without the explicit look-first
            # instruction, models skip the image and go straight to numerics
            # (observed: zero visual references across probe rollouts).
            look_first = (
                "The attached plot(s) are the official rendering of the full time series. "
                "FIRST read them visually and state a hypothesis (is there an anomaly, "
                "roughly where, and at what scale?), THEN verify numerically with the REPL. "
            )
            content = [{"type": "text", "text": f"{look_first}{note}\n\n{sample['question']}"}]
            content += [{"type": "image_url", "image_url": {"url": u}} for u in uris]
            out["prompt"] = [{"role": "user", "content": content}]
        else:
            out["question"] = f"{note}\n\n{sample['question']}"
        return out

    return _render


def _msg_role_content(msg):
    """Read ``role`` and ``content`` from either a dict or a Pydantic Message
    (the RLMEnv loop returns both shapes; verifiers normalizes later)."""
    if hasattr(msg, "role"):
        return msg.role, getattr(msg, "content", None)
    return msg.get("role"), msg.get("content")


def _split_image_tool_message(msg):
    """Workaround for providers that reject image content in ``tool`` role
    messages (e.g. Pinference's Qwen3.5-4B endpoint returns HTTP 400 with
    "Invalid request" when a tool message carries an ``image_url`` part).

    If ``msg`` is a ``tool`` message whose content list contains an image,
    return a 2-element list: [text-only tool message, follow-up user message
    carrying the image]. Otherwise return ``[msg]`` unchanged.

    The original ``tool_call_id`` is preserved on the rewritten tool message
    so the model's history stays well-formed.
    """
    role, content = _msg_role_content(msg)
    if role != "tool" or not isinstance(content, list):
        return [msg]

    text_parts: list[dict] = []
    image_parts: list[dict] = []
    for p in content:
        if isinstance(p, dict):
            t = p.get("type")
        else:
            t = getattr(p, "type", None)
        if t == "text":
            text_parts.append(p if isinstance(p, dict) else p.model_dump())
        elif t == "image_url":
            image_parts.append(p if isinstance(p, dict) else p.model_dump())

    if not image_parts:
        return [msg]

    # Pull tool_call_id / role from the source (dict or Pydantic).
    if hasattr(msg, "model_dump"):
        base = msg.model_dump(exclude_none=True)
    else:
        base = dict(msg)

    new_tool = dict(base)
    new_tool["content"] = (
        text_parts if text_parts
        else [{"type": "text", "text": "[image attached in next turn]"}]
    )

    user_followup = {
        "role": "user",
        "content": [
            {"type": "text", "text": "(image attached for previous tool result)"},
            *image_parts,
        ],
    }
    return [new_tool, user_followup]


class _MultimodalPromptMixin:
    """Strip Arrow-null content keys before sending (OpenAI rejects image
    parts carrying a null ``text`` key — see timeseries_qa._strip_null_content_keys).
    Only matters when the dataset seeds images into the first turn."""

    async def get_model_response(self, state, prompt, **kwargs):
        from timeseries_qa import _strip_null_content_keys
        if isinstance(prompt, list):
            prompt = _strip_null_content_keys(prompt)
        return await super().get_model_response(state, prompt, **kwargs)


class _SeedImageRLMEnv(_MultimodalPromptMixin, RLMEnv):
    """Pure-numeric REPL, but with benchmark images seeded in turn 1."""


class _LookRLMEnv(RLMEnv):
    """RLMEnv with a ``look(path)`` tool that injects images directly into context."""

    async def look(self, path: str, state: State) -> list[dict]:
        """Show an image file inline in the conversation context.

        Call this as a direct tool call after saving a file in call_python_repl.
        The image appears in your next turn — no sub-LLM call is made.

        Args:
            path: Filename in the sandbox working directory (e.g. 'p.png').
        """
        if not _SAFE_PATH.fullmatch(path):
            raise RuntimeError(f"Unsafe path: {path!r}")

        rollout_id = state.get("rollout_id", "")
        session = self._executor._sessions.get(rollout_id)
        sandbox_id = session.sandbox_id if session else None
        if not sandbox_id:
            raise RuntimeError("No sandbox available for look().")

        result = await self._executor._execute_sandbox_command(
            sandbox_id,
            f"base64 -w0 -- {shlex.quote(path)}",
            working_dir=state.get("rlm_fs_root_remote"),
        )
        b64 = (getattr(result, "stdout", "") or "").strip()
        if not b64:
            raise RuntimeError(f"Could not read {path!r} from sandbox.")

        ext = path.rsplit(".", 1)[-1].lower() if "." in path else "png"
        mime = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}" if ext in {"gif", "webp"} else "image/png"
        return [
            {"type": "text", "text": f"[{path}]"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]

    def update_tool_args(self, tool_name: str, tool_args: dict, messages: Messages, state: State, **kwargs) -> dict:
        result = super().update_tool_args(tool_name, tool_args, messages, state, **kwargs) or tool_args
        if tool_name == "look":
            result = dict(result)
            result["state"] = state
        return result

    def __init__(self, *, image_via_user_followup: bool = False,
                 use_local_executor: bool = False, **kwargs):
        """``image_via_user_followup``: when True, post-process the result of
        ``env_response`` so tool messages carrying image content are split
        into (text-only tool) + (follow-up user with image). Needed for
        providers whose chat schema rejects images in ``tool`` role messages
        (Pinference Qwen3.5-4B is the known case as of v1).
        Off by default — the existing gpt-4.1-mini / gpt-5.5 traces stay
        reproducible because no behavior change applies to them."""
        super().__init__(**kwargs)
        self._image_via_user_followup = image_via_user_followup
        if use_local_executor:
            # Swap the Prime-backed executor for the host-local subprocess one.
            # See local_executor.py for the trade-off (no isolation; ~10s/rollout
            # saved on sandbox spin-up). Default False keeps existing reproducibility.
            from local_executor import LocalRLMExecutor
            self._executor = LocalRLMExecutor(self)
        self.add_tool(self.look, args_to_skip=["state"])

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        tool_messages = await super().env_response(messages, state, **kwargs)
        if not self._image_via_user_followup or not tool_messages:
            return tool_messages
        rewritten: list = []
        for m in tool_messages:
            rewritten.extend(_split_image_tool_message(m))
        # The parent override aliases state["final_env_response"] to the
        # unrewritten list; redirect it so any downstream consumer sees the
        # post-split conversation.
        if state.get("final_env_response") is tool_messages:
            state["final_env_response"] = rewritten
        return rewritten


def make_rlm_env(
    ds: TimeSeriesQADataset,
    split: str,
    eval_split: str | None,
    use_image: bool = True,
    enable_sub_llms: bool = False,
    sub_llm_max_turns: int = 5,
    sub_model: str | None = None,
    image_via_user_followup: bool = False,
    use_local_executor: bool = False,
    max_turns: int | None = None,
    seed_images: bool = False,
) -> vf.Environment:
    """Build the RLM env. ``use_image=True`` registers the ``look`` image tool
    (matplotlib + visual inspection); ``use_image=False`` is a pure-numeric REPL.

    ``enable_sub_llms=True`` gives the root model an ``llm_batch()`` tool so it
    can spawn child LLM calls (RLM recursion depth 1 — sub-LLMs themselves are
    leaf-only and cannot recurse further). ``sub_llm_max_turns`` caps each
    child's tool-call loop; ``sub_model`` overrides which model serves them
    (defaults to the same as root).

    ``image_via_user_followup=True`` (use_image only) rewrites each ``look()``
    tool result into [text-only tool message + user-role follow-up that
    carries the image]. Needed for providers whose chat schema rejects
    images inside ``tool`` messages — Pinference's Qwen3.5-4B is the known
    case; gpt-4.1-mini and gpt-5.5 don't need this. Off by default."""

    def correct_answer(completion, answer, state) -> float:
        return ds.score_answer(_final_answer_from(state, completion), answer)

    def length_efficiency(completions, states, answers) -> list[float]:
        """Within-example efficiency nudge (a verifiers *group* reward func — it
        receives the whole GRPO group for one question at once).
        """
        correct = [
            ds.score_answer(_final_answer_from(s, c), a) >= 1.0
            for s, c, a in zip(states, completions, answers)
        ]
        lengths = [float(s.get("root_llm_completion_tokens") or 0) for s in states]
        correct_lengths = [length for length, ok in zip(lengths, correct) if ok]
        if len(correct_lengths) < 2:
            return [0.0] * len(states)
        lo, hi = min(correct_lengths), max(correct_lengths)
        if hi <= lo:
            return [0.0] * len(states)
        return [((hi - length) / (hi - lo)) if ok else 0.0 for length, ok in zip(lengths, correct)]

    LENGTH_BONUS_WEIGHT = 0.1
    metric_funcs = _dataset_metric_funcs_rlm(ds)
    rubric = vf.Rubric(
        funcs=[correct_answer, length_efficiency, *metric_funcs],
        weights=[1.0, LENGTH_BONUS_WEIGHT] + [0.0] * len(metric_funcs),
    )
    nudge = ds.rlm_sub_llm_nudge if enable_sub_llms else ""
    note = _data_note(use_image, ds.rlm_values_note, ds.rlm_answer_format, sub_llm_nudge=nudge,
                      preamble=ds.rlm_task_preamble)
    render = _make_render(note, seed_images=seed_images)
    kwargs = dict(
        dataset=lambda: ds.build(split, render),
        rubric=rubric,
        repl_language="python",
        pip_install_packages="numpy scipy matplotlib" if use_image else "numpy scipy",
        enable_sub_llms=enable_sub_llms,
        sub_llm_max_turns=sub_llm_max_turns,
        # Tar-upload of context.json at L=131k saturates the verifiers default
        # 120s budget under concurrent load (~35% timeout in observed runs);
        # 300s caught most but not all. 900s gives slow workers room.
        max_startup_wait_seconds=900,
    )
    if max_turns is not None:
        # Root-model turn cap (v4: match the training-side RLM_MAX_TURNS=15).
        kwargs["max_turns"] = max_turns
    if enable_sub_llms:
        kwargs["root_prompt_verbosity"] = "heavy"
    if sub_model is not None:
        kwargs["sub_model"] = sub_model
    if eval_split is not None:
        kwargs["eval_dataset"] = lambda: ds.build(eval_split, render)
    if use_image:
        # The flag only matters for the look-image variant; pure-numeric
        # RLMEnv has no image-bearing tool messages to split.
        kwargs["image_via_user_followup"] = image_via_user_followup
        kwargs["use_local_executor"] = use_local_executor
    env_cls = _LookRLMEnv if use_image else (_SeedImageRLMEnv if seed_images else RLMEnv)
    if use_local_executor and not use_image:
        # Numeric-only RLMEnv also benefits from skipping the Prime sandbox.
        from local_executor import LocalRLMExecutor
        env = env_cls(**kwargs)
        env._executor = LocalRLMExecutor(env)
        return env
    return env_cls(**kwargs)
