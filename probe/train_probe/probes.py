"""
probes.py — fit a standardized logistic-regression probe (binary for behavior,
multinomial for strategy/stake), choosing L2 strength by stratified k-fold CV on
the TRAIN split only, then evaluate on the held-out TEST split with bootstrap CIs
for overall accuracy and per-class accuracy.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

import config as C


def fit_probe(X_tr: np.ndarray, y_tr: np.ndarray, multinomial: bool,
              seed: int) -> tuple[Pipeline, float]:
    """Scaler+LogReg pipeline; C chosen by stratified k-fold CV on train.
    Scaler is fit inside CV folds (no leakage). Returns (fitted_pipeline, best_C)."""
    # n_splits can't exceed the smallest class count in train.
    _, class_counts = np.unique(y_tr, return_counts=True)
    n_splits = int(min(C.N_CV_FOLDS, class_counts.min()))
    n_splits = max(n_splits, 2)

    # l2 regularization is sklearn's default
    clf_kwargs = dict(solver="lbfgs", max_iter=C.MAX_ITER)
    try:
        from sklearn.linear_model import LogisticRegression as _LR
        import inspect
        if "multi_class" in inspect.signature(_LR.__init__).parameters:
            clf_kwargs["multi_class"] = (
                "multinomial" if multinomial else "auto")
    except Exception:
        pass

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**clf_kwargs)),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    gs = GridSearchCV(
        pipe, {"clf__C": C.C_GRID}, scoring="accuracy", cv=cv, n_jobs=-1,
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, float(gs.best_params_["clf__C"])


def _per_class_acc(y_true, y_pred, classes):
    out = {}
    for cls in classes:
        m = y_true == cls
        out[cls] = float((y_pred[m] == cls).mean()) if m.any() else float("nan")
    return out


def _bootstrap_cis(y_true, y_pred, classes, seed):
    """Bootstrap over the test set: overall accuracy + per-class accuracy CIs."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    overall, per_cls = [], {c: [] for c in classes}
    for _ in range(C.N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        overall.append((yt == yp).mean())
        for c in classes:
            m = yt == c
            per_cls[c].append((yp[m] == c).mean() if m.any() else np.nan)

    lo, hi = 100 * C.CI_ALPHA / 2, 100 * (1 - C.CI_ALPHA / 2)

    def ci(arr):
        a = np.array(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return (float("nan"), float("nan"))
        return float(np.percentile(a, lo)), float(np.percentile(a, hi))

    return ci(overall), {c: ci(v) for c, v in per_cls.items()}


def evaluate_probe(label_name, pooling, layer,
                   X_tr, y_tr, X_te, y_te, classes, multinomial, seed):
    """Fit on train, evaluate on test, return a flat result dict."""
    model, best_C = fit_probe(X_tr, y_tr, multinomial, seed)
    y_pred = model.predict(X_te)

    overall_acc = float((y_pred == y_te).mean())
    per_class = _per_class_acc(y_te, y_pred, classes)
    overall_ci, per_class_ci = _bootstrap_cis(y_te, y_pred, classes, seed)

    return {
        "label": label_name,
        "pooling": pooling,
        "layer": layer,
        "best_C": best_C,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "accuracy": overall_acc,
        "acc_ci_lo": overall_ci[0],
        "acc_ci_hi": overall_ci[1],
        "per_class_acc": per_class,
        "per_class_ci": per_class_ci,
    }
