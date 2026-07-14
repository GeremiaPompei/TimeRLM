import json

import numpy as np
import pandas as pd
import pytest

_CATEGORIES = (
    ("localize", {"present": True, "start": 10, "end": 20}),
    ("classify_with_evidence", {"kind": "Level Shift", "start": 10, "end": 20}),
    ("measure_magnitude", {"magnitude_sigma": 5.0}),
    ("localize_all_channels", {"anomalies": [{"channel": "ch_00", "start": 10, "end": 20}]}),
    ("lead_lag_with_magnitude", {"direction": "lead", "lag_samples": 100}),
)


@pytest.fixture(scope="session")
def smoke_shard(tmp_path_factory) -> str:
    """A tiny on-disk AnomalyXL shard covering all five categories."""
    length = 1000
    rng = np.random.default_rng(0)
    rows = []
    for i, (category, payload) in enumerate(_CATEGORIES):
        n_channels = 2 if category == "lead_lag_with_magnitude" else 1
        names = (["A_ch_00", "B_ch_00"] if category == "lead_lag_with_magnitude"
                 else ["ch_00"])
        rows.append({
            "id": f"{category}_{i}",
            "category": category,
            "length": length,
            "n_channels": n_channels,
            "question": f"{category} question",
            "answer": f"{category}:" + json.dumps({**payload, "_L": length}),
            "options": [],
            "channel_names": names,
            "channel_values": [rng.normal(size=length).astype(np.float32)
                               for _ in range(n_channels)],
        })
    out = tmp_path_factory.mktemp("anomalyxl_smoke")
    pd.DataFrame(rows).to_parquet(out / "data.parquet")
    return str(out)
