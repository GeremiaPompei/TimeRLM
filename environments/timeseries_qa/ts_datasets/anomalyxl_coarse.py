"""`anomalyxl_coarse` — the AnomalyXL coarse (multiple-choice) shard.

Loads a parquet shard produced by the `anomalyxl` generator (see
`anomalyXL/`). Series are stored in parallel-array
columns (`channel_names`, `channel_values`) so a single parquet can hold
rows with varying channel counts — `to_sample` zips them back into a
`{channel: list[float]}` dict for the renderers.

Scoring mirrors `ARFBench.score_answer`: JSON `{"answer": ...}` first,
`Answer:` prefix second, strip-equality fallback. Filtering knobs let users
sub-select (`length_max`, `n_channels_max`, `categories`) without
regenerating the parquet.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE

from .base import Sample, TimeSeriesQADataset, strip_reasoning

# Precise-localization rows carry their gold as "{index}@{L}" so the scorer
# can apply tolerance-of-L bands without re-looking-up L. See the
# precise_{start,end}_time question templates.
_PRECISE_GOLD_RE = re.compile(r"^\s*(-?\d+)\s*@\s*(\d+)\s*$")
# Non-negative integer not adjacent to a word char or `@`, and not a
# percentage commentary (`57%`). Excludes the L token of an echoed gold
# (`73214@128000`) — that path is handled separately, below.
_INT_RE = re.compile(r"(?<![\w@])(\d{1,12})(?!\d|@|\s*%)")

# Categories whose gold is `"{idx}@{L}"`. Used to decide whether precise-
# localization metrics make sense for an eval (see AnomalyXLCoarse.__init__).
_PRECISE_CATEGORIES: frozenset[str] = frozenset({"precise_start_time", "precise_end_time"})


def is_precise_gold(gold: str) -> bool:
    return bool(_PRECISE_GOLD_RE.match(gold or ""))


def parse_precise_gold(gold: str) -> tuple[int, int]:
    """Return ``(t_idx, L)`` from ``"{idx}@{L}"``. Raises if the gold is malformed."""
    m = _PRECISE_GOLD_RE.match(gold or "")
    if not m:
        raise ValueError(f"Not a precise gold: {gold!r}")
    return int(m.group(1)), int(m.group(2))


def parse_precise_pred(text: str) -> int | None:
    """Extract a non-negative integer prediction from model text. Returns
    ``None`` when nothing parseable is present.

    Resolution order:
      1. Echoed gold ``"{idx}@{L}"`` — recovers correctness when the model
         copies the gold format verbatim.
      2. JSON ``{"answer": ...}`` blob.
      3. Last non-negative integer in the text — final answers typically
         appear at the end of model prose, and the leading-integer fallback
         picked up stray ints from setups like ``"0 < t_start < 73214"``.
    """
    if not text:
        return None
    s = text.strip()
    m = _PRECISE_GOLD_RE.search(s)
    if m:
        return int(m.group(1))
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            val = obj.get("answer")
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str):
                inner = val.strip()
                m2 = _PRECISE_GOLD_RE.search(inner)
                if m2:
                    return int(m2.group(1))
                ints = _INT_RE.findall(inner)
                if ints:
                    return int(ints[-1])
    ints = _INT_RE.findall(s)
    if ints:
        return int(ints[-1])
    return None


# Default points at the MCQ (coarse) test split of the released AnomalyXL monorepo.
# Override with the `ANOMALYXL_COARSE_DATA_ROOT` env var, by passing `repo_or_path=...`
# (e.g. a locally generated dir from `anomalyXL/scripts/coarse_project.py`), or by
# selecting another split (`nz00shuuuu/AnomalyXL#MCQ/{train,val,test}`).
_DEFAULT_REPO_OR_PATH = "nz00shuuuu/AnomalyXL#MCQ/test"


def _resolve_path(repo_or_path: str) -> Path:
    """Resolve `repo_or_path` to a local directory containing `data.parquet`.

    A literal local path wins (even if it happens to look like a repo id).
    Otherwise it is treated as a Hugging Face dataset repo, optionally with a
    ``#fragment`` selecting what to pull out of it. The released layouts are
    understood:

    ``<repo>``                    root-level ``data.parquet`` (+ ``manifest.json``)
    ``<repo>#<subdir>``           ``<subdir>/data.parquet`` — e.g.
                                  ``nz00shuuuu/anomalyxl-generalization#ltaf-anomaly``
    ``<repo>#<path>``             a parquet named exactly ``<path>.parquet`` — the
                                  partition/split layout of the ``nz00shuuuu/AnomalyXL``
                                  monorepo, e.g. ``nz00shuuuu/AnomalyXL#Localize/test``
                                  or ``nz00shuuuu/AnomalyXL#MCQ/test``.
    ``<repo>#<split>``            ``data/<split>.parquet`` (a flat train/validation/test
                                  layout). Defaults to ``test`` when no fragment is given.

    The selected files are linked into a small cache directory named
    ``data.parquet`` (+ ``manifest.json`` when the repo ships one), which is what
    the loaders below expect. A warm cache is returned WITHOUT touching the network,
    so offline runs work and the RL orchestrator's many env workers do not each hit
    the Hub API.
    """
    spec, _, frag = repo_or_path.partition("#")
    p = Path(spec).expanduser()
    if p.exists():
        if p.is_file():
            raise NotADirectoryError(
                f"repo_or_path must be a DIRECTORY containing data.parquet, got the "
                f"file {p}. Pass its parent directory instead."
            )
        return p
    # A Hub repo id is `namespace/name` — at most one slash, no path sigils. Anything
    # else is a filesystem path that does not exist: say so, instead of handing it to
    # the Hub client and surfacing an opaque HFValidationError.
    if spec.startswith(("/", ".", "~")) or spec.count("/") > 1:
        raise FileNotFoundError(f"No such directory: {spec}")

    cache = Path(HF_HUB_CACHE) / "timerlm" / spec.replace("/", "__") / (frag or "_default")
    parquet = cache / "data.parquet"
    if parquet.exists():
        return cache  # warm cache — no network

    files = set(HfApi().list_repo_files(spec, repo_type="dataset"))

    # Pick the layout: subdir/data.parquet, <path>.parquet, root data.parquet, or
    # data/<split>.parquet.
    if frag and f"{frag}/data.parquet" in files:
        remote_parquet = f"{frag}/data.parquet"
    elif frag and f"{frag}.parquet" in files:
        remote_parquet = f"{frag}.parquet"
    elif not frag and "data.parquet" in files:
        remote_parquet = "data.parquet"
    else:
        split = frag or "test"
        remote_parquet = f"data/{split}.parquet"
        if remote_parquet not in files:
            raise FileNotFoundError(
                f"{spec!r} has no '{frag or 'data'}' shard. Looked for "
                f"'{frag}/data.parquet', '{frag}.parquet', 'data.parquet', and "
                f"'{remote_parquet}'. Available: "
                f"{sorted(f for f in files if f.endswith(('.parquet', '.json')))}"
            )

    # Manifest ships next to the parquet (partition dir) or at the repo root, if at all.
    parquet_dir = remote_parquet.rsplit("/", 1)[0] if "/" in remote_parquet else ""
    remote_manifest = next(
        (m for m in (f"{parquet_dir}/manifest.json" if parquet_dir else None, "manifest.json")
         if m and m in files),
        None,
    )

    cache.mkdir(parents=True, exist_ok=True)
    for remote, local in [(remote_parquet, "data.parquet"), (remote_manifest, "manifest.json")]:
        if remote is None:
            continue
        dst = cache / local
        src = Path(hf_hub_download(spec, remote, repo_type="dataset"))
        # Link atomically: two workers resolving the same shard on a cold cache race
        # here, and the loser must not blow up. os.replace() over a uniquely-named
        # temp link is atomic on POSIX; fall back to a copy where symlinks are not
        # permitted (e.g. Windows without developer mode).
        tmp = cache / f".{local}.{os.getpid()}.tmp"
        try:
            tmp.symlink_to(src)
            os.replace(tmp, dst)
        except OSError:
            tmp.unlink(missing_ok=True)
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except shutil.SameFileError:
                    pass
    return cache


class AnomalyXLCoarse(TimeSeriesQADataset):
    name = "anomalyxl_coarse"
    splits = ("train", "eval")
    default_split = "train"
    default_eval_split = "eval"
    rlm_values_note = (
        "Channels in paired (Tier III) questions are prefixed with `A_` or `B_`. "
    )
    rlm_answer_format = (
        "set answer['content'] to the chosen option exactly as it appears in the "
        "options list (copy the text verbatim), then set answer['ready'] = True."
    )
    rlm_sub_llm_nudge = (
        "Series can be up to ~131k samples; for questions that benefit from "
        "narrative description of multiple sections (e.g. comparing what happens "
        "in the first quarter vs the last quarter, or characterizing each of several "
        "windows), prefer `llm_batch` over a single mega-prompt: split the series "
        "into segments, save each segment's plot or summary stats to a file, then "
        "send a batch of section-level questions to sub-LLMs in parallel. Use it "
        "when you need semantic judgment per section, not when numpy stats suffice. "
    )

    def __init__(
        self,
        repo_or_path: str | None = None,
        eval_size: int = 200,
        seed: int = 42,
        length_min: int | None = None,
        length_max: int | None = None,
        n_channels_max: int | None = None,
        categories: tuple[str, ...] | list[str] | None = None,
        ids: tuple[str, ...] | list[str] | None = None,
        mode: str = "text",
        text_max_samples_per_channel: int = 8192,
        text_total_samples: int | None = None,
        image_max_samples_per_channel: int = 16384,
    ):
        """`repo_or_path`: local directory or HF dataset repo id holding
        `data.parquet`. When unset, falls back to `$ANOMALYXL_COARSE_DATA_ROOT` and
        then to the pilot shard path (`_DEFAULT_REPO_OR_PATH`).
        `length_max` / `n_channels_max` filter the in-memory frame;
        `categories` keeps only those question categories.

        `text_max_samples_per_channel` decimates each channel for text mode so
        an `L=128k` series doesn't emit ~800k chars per sample (well past any
        LLM window). This is the realistic text-mode behavior at long L: the
        model sees a coarsened view and the anomaly (absolute width ≤256) may
        be smeared across <1 retained sample. `image_max_samples_per_channel`
        does the same for plot legibility. RLM mode bypasses both — the model
        reads the full series from a sandbox file.

        `text_total_samples` (optional) caps the *total* sample count summed
        across all channels in a text-mode prompt. When set, the per-channel
        budget becomes ``max(256, text_total_samples // n_channels)`` so a
        4-channel row at L=131k doesn't emit 4× the single-channel budget and
        overflow Qwen-class 32k input windows. Ignored on `image`/`rlm`."""
        self.repo_or_path = (
            repo_or_path
            or os.environ.get("ANOMALYXL_COARSE_DATA_ROOT")
            or _DEFAULT_REPO_OR_PATH
        )
        self.eval_size = eval_size
        self.seed = seed
        self.length_min = length_min
        self.length_max = length_max
        self.n_channels_max = n_channels_max
        self.categories = tuple(categories) if categories else None
        # `ids` lets callers pick a specific row subset (e.g., the errored rows
        # from a prior run) without writing a temp parquet. Applied alongside
        # other filters, before the seed shuffle.
        self.ids = tuple(ids) if ids else None
        self.mode = mode
        self.text_max_samples_per_channel = text_max_samples_per_channel
        self.text_total_samples = text_total_samples
        self.image_max_samples_per_channel = image_max_samples_per_channel
        # Gate precise-localization metrics on the eval being filtered to
        # precise categories only. Verifiers' rubric/EnvMetrics averages
        # metrics over every row without per-row skipping, so reporting
        # `acc_at_1pct` on a mixed eval would dilute the average with zeros
        # from non-precise rows (50 precise hits out of 50 + 50 categorization
        # rows reads as 0.50, not 1.00). Restrict to homogeneous slices —
        # users wanting precise metrics pass `categories=["precise_*"]`.
        precise_only = (
            self.categories is not None
            and len(self.categories) > 0
            and all(c in _PRECISE_CATEGORIES for c in self.categories)
        )
        self.extra_metric_names = (
            tuple(self._band_name(b) for b in self.PRECISE_ACC_BANDS)
            + ("mae_samples", "mae_fraction")
        ) if precise_only else ()
        self._frame: pd.DataFrame | None = None
        self._train_df: pd.DataFrame | None = None
        self._eval_df: pd.DataFrame | None = None

    def _load_frame(self) -> pd.DataFrame:
        if self._frame is not None:
            return self._frame
        root = _resolve_path(self.repo_or_path)
        parquet = root / "data.parquet"
        if not parquet.exists():
            raise FileNotFoundError(f"No data.parquet under {root}")
        df = pd.read_parquet(parquet)
        if self.length_min is not None:
            df = df[df["length"] >= int(self.length_min)]
        if self.length_max is not None:
            df = df[df["length"] <= int(self.length_max)]
        if self.n_channels_max is not None:
            df = df[df["n_channels"] <= int(self.n_channels_max)]
        if self.categories is not None:
            df = df[df["category"].isin(self.categories)]
        if self.ids is not None:
            df = df[df["id"].isin(self.ids)]
        df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        self._frame = df
        return df

    def _split_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._train_df is not None and self._eval_df is not None:
            return self._train_df, self._eval_df
        df = self._load_frame()
        n_eval = min(self.eval_size, len(df))
        self._eval_df = df.iloc[:n_eval].reset_index(drop=True)
        self._train_df = df.iloc[n_eval:].reset_index(drop=True)
        return self._train_df, self._eval_df

    def load_split(self, split: str) -> Dataset:
        train, ev = self._split_dfs()
        return Dataset.from_pandas(ev if split == "eval" else train, preserve_index=False)

    # Channel values are stored as float32 rounded to 3 decimals at parquet
    # write time (see `anomalyxl/cli.py:_VALUE_DECIMALS`). The float32 round-trip
    # to Python float adds trailing-bit noise that bloats `str(v)`; snap back
    # to 3 decimals here so text-mode rendering stays compact.
    _VALUE_DECIMALS = 3

    def to_sample(self, row: dict) -> Sample:
        names = list(row["channel_names"])
        values = [list(v) for v in row["channel_values"]]
        series = {n: [round(float(x), self._VALUE_DECIMALS) for x in v] for n, v in zip(names, values)}
        original_length = int(row["length"])
        # Decimate per-mode so long-L prompts don't explode and plots stay
        # legible. RLM mode keeps the full series — the model reads it from
        # a sandbox file, not from the chat context.
        cap = self._decimation_cap(n_channels=len(names))
        if cap is not None and original_length > cap:
            step = max(1, original_length // cap)
            series = {n: v[::step] for n, v in series.items()}
        options = list(row["options"])
        # Per-row deterministic shuffle so option order is stable across
        # reloads but un-leakable across rows.
        if options:
            rng = random.Random(f"{self.seed}:{row.get('id', '')}")
            rng.shuffle(options)
            options_str = json.dumps(options, ensure_ascii=False)
            question_text = f"{row['question']}\nOptions: {options_str}"
        else:
            # Precise/open-response categories: no options block at all.
            question_text = row["question"]
        return {
            "series": series,
            "question": question_text,
            "answer": str(row["answer"]),
            "length": original_length,
        }

    def _decimation_cap(self, n_channels: int = 1) -> int | None:
        if self.mode == "text":
            # When the total-budget knob is set, distribute the budget across
            # channels with a floor at 256 samples/channel. Falls back to the
            # per-channel cap for single-channel runs or when the knob is unset.
            if self.text_total_samples is not None and n_channels > 0:
                per_channel = max(256, self.text_total_samples // n_channels)
                return min(self.text_max_samples_per_channel, per_channel)
            return self.text_max_samples_per_channel
        if self.mode == "image":
            return self.image_max_samples_per_channel
        return None  # RLM mode: full series via file

    # Accuracy thresholds reported for precise-localization rows, as fractions
    # of L. The first entry doubles as the primary reward threshold in
    # `score_answer`.
    PRECISE_ACC_BANDS: tuple[float, ...] = (0.01, 0.001, 0.005, 0.05)

    @staticmethod
    def _band_name(b: float) -> str:
        """Stable metric name for tolerance band ``b`` (fraction of L).

        ``0.01 -> "acc_at_1pct"``, ``0.001 -> "acc_at_0p1pct"``. The ``p``
        substitution keeps the name a valid identifier for metric-dict keys.
        """
        return f"acc_at_{b*100:g}pct".replace(".", "p")

    def score_answer(self, parsed_text: str, gold: str) -> float:
        gold = (gold or "").strip()
        if is_precise_gold(gold):
            true_idx, L = parse_precise_gold(gold)
            pred_idx = parse_precise_pred(parsed_text or "")
            if pred_idx is None or L == 0:
                return 0.0
            primary_tol = self.PRECISE_ACC_BANDS[0]
            return 1.0 if abs(pred_idx - true_idx) / L <= primary_tol else 0.0
        for candidate in self._answer_candidates(parsed_text or ""):
            if candidate.strip() == gold:
                return 1.0
        return 0.0

    def score_metrics(self, parsed_text: str, gold: str) -> dict[str, float]:
        # Gated by `__init__`: `extra_metric_names` is empty unless the eval is
        # filtered to precise categories. When empty, the env never asks for
        # these, so returning `{}` is the correct no-op.
        if not self.extra_metric_names:
            return {}
        gold = (gold or "").strip()
        if not is_precise_gold(gold):
            # Defensive: precise-only filter should keep these rows out, but
            # keep the metric column well-defined if one slips through.
            return {name: 0.0 for name in self.extra_metric_names}
        true_idx, L = parse_precise_gold(gold)
        pred = parse_precise_pred(parsed_text or "")
        out: dict[str, float] = {}
        for b in self.PRECISE_ACC_BANDS:
            name = self._band_name(b)
            if pred is None or L == 0:
                out[name] = 0.0
            else:
                out[name] = 1.0 if abs(pred - true_idx) / L <= b else 0.0
        out["mae_samples"] = float(abs(pred - true_idx)) if pred is not None else float(L)
        out["mae_fraction"] = (abs(pred - true_idx) / L) if (pred is not None and L > 0) else 1.0
        return out

    @staticmethod
    def _answer_candidates(text: str):
        # Drop <think>...</think> first: reasoning models emit the answer after
        # it, and a bare MCQ answer on the final line would otherwise be missed.
        text = strip_reasoning(text or "")
        yield text
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            try:
                obj = json.loads(text[start:end])
            except (json.JSONDecodeError, TypeError, ValueError):
                obj = None
            if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
                yield obj["answer"]
        m = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", text)
        if m:
            yield m.group(1).strip().strip("\"'")
        # The bare final-line answer (e.g. Qwen ends "</think>\n\nYes").
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            yield lines[-1].strip("\"'")
