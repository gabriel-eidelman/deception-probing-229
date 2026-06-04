"""
data_loading.py — load the frozen grader labels from scored_responses.csv and
the layerwise activation matrices from Phase B, aligned row-for-row by
response_id.

The single coupling point with Phase B is the activation file scheme described
in config.py.  If Phase B saved activations differently (e.g. one big .npz, or
torch .pt), reimplement load_layer_matrix() / load_response_ids() / detect_layers()
here and nothing else changes.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

import config as C


# ──────────────────────────────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────────────────────────────
def load_labels() -> pd.DataFrame:
    """Return a DataFrame indexed by response_id with one column per label
    ('behavior', 'strategy', 'stake_structure'), values as class strings."""
    df = pd.read_csv(C.SCORED_CSV)

    if "response_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "response_id"})

    out = pd.DataFrame({"response_id": df["response_id"].astype(int)})

    # behavior — frozen grader label
    out["behavior"] = df[C.BEHAVIOR_COL].astype(str).str.lower()

    # strategy — elicitation strategy
    out["strategy"] = df[C.STRATEGY_COL].astype(str)

    # stake_structure — collapse to 3 design classes
    if C.STAKE_COL in df.columns:
        out["stake_structure"] = df[C.STAKE_COL].astype(str).map(C.collapse_stake)
    else:
        raise KeyError(
            f"'{C.STAKE_COL}' not in {C.SCORED_CSV}. Columns: {list(df.columns)}"
        )

    out = out.set_index("response_id").sort_index()
    _validate_labels(out)
    return out


def _validate_labels(df: pd.DataFrame) -> None:
    for name, spec in C.LABELS.items():
        present = set(df[name].unique())
        unexpected = present - set(spec["classes"])
        if unexpected:
            print(f"  [warn] label '{name}' has values outside the expected "
                  f"classes {spec['classes']}: {sorted(unexpected)}")
        counts = df[name].value_counts().to_dict()
        print(f"  {name}: {counts}")


# ──────────────────────────────────────────────────────────────────────
# Activations
# ──────────────────────────────────────────────────────────────────────
def detect_layers() -> list[int]:
    """Auto-detect available layers from files like 'last_token_layer{L}.npy'.
    Honors config.LAYERS if set."""
    if C.LAYERS is not None:
        return list(C.LAYERS)
    pat = re.compile(r"^(?:last_token|mean_pooled)_layer(\d+)\.npy$")
    layers = set()
    for p in C.ACTIVATIONS_DIR.glob("*_layer*.npy"):
        m = pat.match(p.name)
        if m:
            layers.add(int(m.group(1)))
    if not layers:
        raise FileNotFoundError(
            f"No activation files matching '*_layer*.npy' in {C.ACTIVATIONS_DIR}. "
            "Set config.LAYERS / config.act_path to match Phase B's output."
        )
    return sorted(layers)


def load_response_ids() -> np.ndarray:
    """Row order of the activation matrices, as response_ids aligning to the CSV."""
    if not C.RESPONSE_IDS_PATH.exists():
        raise FileNotFoundError(
            f"{C.RESPONSE_IDS_PATH} missing. Phase B must save the response_id "
            "for each activation row so activations align to grader labels."
        )
    return np.load(C.RESPONSE_IDS_PATH).astype(int)


def load_layer_matrix(pooling: str, layer: int) -> np.ndarray:
    """[n_examples, d_model] activations for one pooling method at one layer."""
    p = C.act_path(pooling, layer)
    if not p.exists():
        raise FileNotFoundError(f"Missing activation file: {p}")
    X = np.load(p)
    if X.ndim != 2:
        raise ValueError(f"{p} has shape {X.shape}; expected [n_examples, d_model]")
    return X.astype(np.float64)


def aligned_labels(response_ids: np.ndarray, labels_df: pd.DataFrame,
                   label_name: str) -> np.ndarray:
    """Return label values for each activation row, in row order."""
    missing = set(response_ids) - set(labels_df.index)
    if missing:
        raise KeyError(
            f"{len(missing)} activation response_ids absent from labels "
            f"(e.g. {sorted(missing)[:5]}). Activations/labels are misaligned."
        )
    return labels_df.loc[response_ids, label_name].to_numpy()
