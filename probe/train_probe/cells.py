"""Shared cell masks and peak-layer loading for transfer, orthogonalization, and geometry runs."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

import config as C
import data_loading as D
import probes as P


# ──────────────────────────────────────────────────────────────────────
# Path-A cell definitions (from the Updated Plan, Phase A.1).
#   TRAIN: role_based  x prosocial
#   TEST : instructed  x self_serving   AND   instrumental x self_serving
# A "cell" here is a (strategy, stake_structure) pair. Behavior is the label
# we probe for; strategy/stake define which cell a row belongs to.
# ──────────────────────────────────────────────────────────────────────
TRAIN_CELL = {"strategy": "role_based", "stake_structure": "prosocial"}
TEST_CELLS = [
    {"strategy": "instructed", "stake_structure": "self_serving"},
    {"strategy": "instrumental", "stake_structure": "self_serving"},
]


def cell_mask(response_ids: np.ndarray, labels_df: pd.DataFrame,
              cell: dict) -> np.ndarray:
    """Boolean mask over activation row order for rows in `cell`.

    Matches on strategy and stake_structure exactly (after the same
    collapse_stake() normalization C.1 applied via load_labels)."""
    strat = D.aligned_labels(response_ids, labels_df, "strategy")
    stake = D.aligned_labels(response_ids, labels_df, "stake_structure")
    return (strat == cell["strategy"]) & (stake == cell["stake_structure"])


def behavior_labels(response_ids: np.ndarray,
                    labels_df: pd.DataFrame) -> np.ndarray:
    """The frozen grader behavior label, in activation row order."""
    return D.aligned_labels(response_ids, labels_df, "behavior")


def describe_mask(name: str, mask: np.ndarray,
                  y: np.ndarray) -> dict:
    """Per-class counts for a masked set; printed and stored so census
    imbalance is visible in every result file."""
    sub = y[mask]
    classes, counts = np.unique(sub, return_counts=True)
    d = {str(c): int(n) for c, n in zip(classes, counts)}
    print(f"    {name:32} n={int(mask.sum()):4d}  {d}")
    return {"name": name, "n": int(mask.sum()), "per_class": d}


# ──────────────────────────────────────────────────────────────────────
# Peak layer chosen by C.1.
# ──────────────────────────────────────────────────────────────────────
def load_peak() -> dict:
    if not C.PEAK_LAYER_PATH.exists():
        raise FileNotFoundError(
            f"{C.PEAK_LAYER_PATH} missing. Run run_pipeline.py (C.1) first — "
            "it selects the peak behavior layer that C.2/C.3/D.3 run at."
        )
    with open(C.PEAK_LAYER_PATH) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# Fit + evaluate a behavior probe on arbitrary train/test masks.
# Thin wrapper over probes.evaluate_probe so accuracy + CIs use the exact
# same code path (and therefore the same convention) as C.1.
# ──────────────────────────────────────────────────────────────────────
def fit_eval_behavior(X, y, train_mask, test_mask, seed=None):
    """Returns the probes.evaluate_probe result dict for a behavior probe
    trained on train_mask rows and tested on test_mask rows."""
    seed = C.SEED if seed is None else seed
    return P.evaluate_probe(
        label_name="behavior", pooling="(cell)", layer=-1,
        X_tr=X[train_mask], y_tr=y[train_mask],
        X_te=X[test_mask], y_te=y[test_mask],
        classes=C.BEHAVIOR_CLASSES, multinomial=False, seed=seed,
    )


def fitted_behavior_probe(X, y, train_mask, seed=None):
    """Just the fitted (scaler+logreg) pipeline + best_C, for geometry (D.3)
    where we want the weight vector, not a transfer accuracy."""
    seed = C.SEED if seed is None else seed
    return P.fit_probe(X[train_mask], y[train_mask], multinomial=False,
                       seed=seed)


# ──────────────────────────────────────────────────────────────────────
# Bootstrap CI on a single accuracy number, matching probes.py convention
# (percentile bootstrap over the test set at CI_ALPHA).
# Used by D.2 (MLP) and anywhere we score predictions outside evaluate_probe.
# ──────────────────────────────────────────────────────────────────────
def accuracy_ci(y_true, y_pred, seed=None):
    seed = C.SEED if seed is None else seed
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boots = []
    for _ in range(C.N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        boots.append((y_true[idx] == y_pred[idx]).mean())
    lo, hi = 100 * C.CI_ALPHA / 2, 100 * (1 - C.CI_ALPHA / 2)
    return (float(np.percentile(boots, lo)),
            float(np.percentile(boots, hi)))


# ══════════════════════════════════════════════════════════════════════
# C.3 orthogonalization: estimate the "combined protocol direction".
#
# The plan is explicit that this removes ONE combined linear direction
# (strategy and stake are partially collinear, so they are not two
# separable factors). We expose three estimators of that single direction
# and report the behavior probe under each, with the logistic-probe normal
# as the headline and the other two as robustness rows.
#
# All directions are estimated on the SAME data the behavior probe trains on
# (the train rows), standardized exactly as the probe standardizes, then the
# direction is projected out of BOTH train and test before the behavior probe
# is refit. Returns a UNIT vector in the standardized feature space.
# ══════════════════════════════════════════════════════════════════════
def safe_scaler(X_fit):
    """StandardScaler that won't emit NaNs: zero-variance columns (constant
    features, which occur in small samples) keep scale=1 instead of 0. Fit on
    X_fit, returned ready to .transform()."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_fit)
    sc.scale_ = np.where(sc.scale_ == 0, 1.0, sc.scale_)
    return sc


def make_logreg(multinomial: bool, **overrides):
    """Construct a LogisticRegression, passing multi_class only on sklearn
    versions that still accept it (removed in >=1.8, where multinomial is the
    default for multiclass anyway). Mirrors the guard in probes.fit_probe so
    every logistic fit in the project behaves identically across versions."""
    from sklearn.linear_model import LogisticRegression
    import inspect
    kw = dict(solver="lbfgs", max_iter=C.MAX_ITER, C=1.0)
    kw.update(overrides)
    if "multi_class" in inspect.signature(LogisticRegression.__init__).parameters:
        kw["multi_class"] = "multinomial" if multinomial else "auto"
    return LogisticRegression(**kw)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-norm protocol direction")
    return v / n


def protocol_direction_logistic(Xs, protocol_labels, seed):
    """Headline: normal of a multinomial logistic probe trained to predict the
    combined protocol label (the cell identity). For a multiclass probe we take
    the first principal component of the coefficient rows so we still remove a
    SINGLE direction, per the plan's 'one combined direction' framing.

    Xs: already-standardized features (train rows).
    protocol_labels: string cell id per train row.
    """
    from sklearn.linear_model import LogisticRegression
    clf = make_logreg(multinomial=True)
    clf.fit(Xs, protocol_labels)
    W = np.atleast_2d(clf.coef_)            # [n_classes_or_1, d]
    if W.shape[0] == 1:
        return _unit(W[0])
    # Multiclass: collapse the coefficient rows to their leading direction.
    # Center the rows then take the top right-singular vector.
    Wc = W - W.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
    return _unit(Vt[0])


def protocol_direction_diffmeans(Xs, in_train_cell_mask):
    """Difference of class means: mean(TRAIN cell) - mean(rest), in
    standardized space. The most assumption-light single direction.

    in_train_cell_mask: boolean over the train rows, True = role_basedxprosocial.
    """
    a = Xs[in_train_cell_mask].mean(axis=0)
    b = Xs[~in_train_cell_mask].mean(axis=0)
    return _unit(a - b)


def protocol_direction_pca(Xs, protocol_labels):
    """Top PCA direction of the class-mean matrix (between-cell scatter):
    the single axis along which the protocol cells most separate.
    """
    cells = sorted(set(protocol_labels))
    means = np.stack([Xs[np.array(protocol_labels) == c].mean(axis=0)
                      for c in cells])
    means_c = means - means.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(means_c, full_matrices=False)
    return _unit(Vt[0])


def project_out(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the component of every row along `direction` (unit vector).
        X_res = X - (X @ d) d^T
    Operates in whatever space X is in; pass standardized X and a direction
    estimated in the same standardized space."""
    d = _unit(direction)
    return X - np.outer(X @ d, d)
