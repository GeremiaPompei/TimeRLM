"""Present a series to the model: as text, or as a plotted image."""

from __future__ import annotations

import base64
import io
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Series = Mapping[str, Sequence[float]]


def format_timeseries(series: Series) -> str:
    """One ``name: v1, v2, ...`` line per channel."""
    return "\n".join(f"{name}: " + ", ".join(str(v) for v in values) for name, values in series.items())


def plot_timeseries(series: Series, *, length: int | None = None) -> str:
    """Plot the channels and return a base64 PNG ``data:`` URI.

    ``length`` is the *original* (pre-decimation) sample count. When the
    caller has decimated the series for plot legibility, pass it so the
    x-axis still reads in the original sample indices — otherwise the
    model sees a plot that disagrees with the gold's "report the
    integer sample index" instruction. When ``length`` is None the
    axis falls back to the decimated index range.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    for name, values in series.items():
        values = list(values)
        if length is not None and len(values) > 0:
            # Plot decimated values at their original x-positions so the
            # axis label still names true sample indices.
            n = len(values)
            xs = [int(round(i * (length - 1) / max(1, n - 1))) for i in range(n)]
        else:
            xs = list(range(len(values)))
        ax.plot(xs, values, label=name, linewidth=1)
    if len(series) > 1:
        ax.legend(loc="best", fontsize="small")
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
