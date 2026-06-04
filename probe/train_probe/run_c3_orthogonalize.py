"""
run_c3_orthogonalize.py — C.3 (orthogonalization), the decisive causal test.

Run AFTER run_pipeline.py (C.1), from probe/train_probe:
    python run_c3_orthogonalize.py

Procedure, at C.1's peak layer, per pooling method:
  1. Standardize features (fit scaler on the in-distribution train rows only).
  2. Estimate the SINGLE combined protocol direction in standardized space.
     The plan is explicit this is ONE combined direction, not two separable
     factors (strategy/stake are partially collinear). We estimate it three
     ways and report all three:
        logistic  (headline) — leading direction of a protocol-classifier
        diffmeans (robustness) — mean(train cell) - mean(rest)
        pca       (robustness) — top axis of between-cell scatter
  3. Project that direction out of BOTH train and test features.
  4. Refit the behavior probe on the residual; compare accuracy before vs after.

     Survives orthogonalization -> evidence for a deception representation not
                                    reducible to the protocol.
     Vanishes (-> majority/chance) -> the behavior signal WAS the protocol
                                    signal in costume.

This is evaluated two ways for the held-out test set:
  (a) IN-DISTRIBUTION: the C.1 persisted random split (same rows C.1 scored
      behavior on) — "does behavior survive within the training distribution?"
  (b) TRANSFER: train cell -> self_serving cells (the C.2 split) — "does what
      survives also transfer?"  This is the strongest single number.

LIMITATIONS baked into the output (D.1): removal is LINEAR and removes ONE
combined direction. A nonlinear residual could remain — that's what D.2 tests.

Produces under ./outputs/:
    c3_orthogonalize.csv    rows: pooling x eval_mode x direction x {pre,post}
    c3_orthogonalize.json   full results incl. the residual variance removed
"""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

import config as C
import data_loading as D
import split as S
import cells as CL

DIRECTION_ESTIMATORS = ["logistic", "diffmeans", "pca"]
HEADLINE = "logistic"


def _protocol_label(response_ids, labels_df):
    """Combined protocol (cell) id per row: 'strategy|stake'."""
    strat = D.aligned_labels(response_ids, labels_df, "strategy")
    stake = D.aligned_labels(response_ids, labels_df, "stake_structure")
    return np.array([f"{s}|{k}" for s, k in zip(strat, stake)])


def _fit_logreg(Xs, y, multinomial):
    clf = CL.make_logreg(multinomial=multinomial)
    clf.fit(Xs, y)
    return clf


def _score(Xs_tr, y_tr, Xs_te, y_te, seed):
    """Fit a behavior logreg on standardized train, return acc + bootstrap CI
    on test, using cells.accuracy_ci (same convention as C.1)."""
    clf = _fit_logreg(Xs_tr, y_tr, multinomial=False)
    pred = clf.predict(Xs_te)
    acc = float((pred == y_te).mean())
    lo, hi = CL.accuracy_ci(y_te, pred, seed)
    return acc, lo, hi, _majority_chance(y_te)


def _majority_chance(y_sub):
    if len(y_sub) == 0:
        return float("nan")
    _, counts = np.unique(y_sub, return_counts=True)
    return float(counts.max() / counts.sum())


def _direction(kind, Xs_tr, protocol_tr, train_cell_local):
    if kind == "logistic":
        return CL.protocol_direction_logistic(Xs_tr, protocol_tr, C.SEED)
    if kind == "diffmeans":
        return CL.protocol_direction_diffmeans(Xs_tr, train_cell_local)
    if kind == "pca":
        return CL.protocol_direction_pca(Xs_tr, protocol_tr)
    raise ValueError(kind)


def _variance_along(Xs, d):
    """Fraction of total variance lying along unit direction d."""
    proj = Xs @ d
    return float(proj.var() / (Xs.var(axis=0).sum() + 1e-12))


def _run_one_eval(eval_name, X, y, protocol, train_mask, test_mask, seed):
    """Standardize on train rows, then pre/post-orthogonalization scores for
    every direction estimator. Returns (rows, detail).

    The protocol direction is estimated on the standardized TRAIN+TEST union,
    not the train rows alone: in transfer mode the train cell is single-protocol
    (e.g. role_based|prosocial only), so the protocol direction is undefined
    within it. Orthogonalization removes a geometric axis from the feature
    space; that axis is defined wherever protocol varies, independent of which
    rows train the behavior probe. We fit ONE scaler (on train) and apply it to
    both, so train/test live in the same standardized space as the direction."""
    scaler = CL.safe_scaler(X[train_mask])
    Xs_tr = scaler.transform(X[train_mask])
    Xs_te = scaler.transform(X[test_mask])
    y_tr, y_te = y[train_mask], y[test_mask]
    protocol_tr = protocol[train_mask]

    # Direction-fitting set: union of train+test in the SAME standardized space.
    fit_mask = train_mask | test_mask
    Xs_fit = scaler.transform(X[fit_mask])
    protocol_fit = protocol[fit_mask]

    # which fit rows are the TRAIN cell (for diffmeans)
    tc = CL.TRAIN_CELL
    train_cell_id = f'{tc["strategy"]}|{tc["stake_structure"]}'
    train_cell_fit = (protocol_fit == train_cell_id)
    if train_cell_fit.sum() == 0 or train_cell_fit.all():
        # in-distribution eval (no single TRAIN cell gating); diffmeans falls
        # back to most- vs least-frequent protocol cell so it still removes a
        # protocol-aligned axis.
        top = pd.Series(protocol_fit).value_counts().index[0]
        train_cell_fit = (protocol_fit == top)

    single_protocol_fit = len(set(protocol_fit)) < 2

    pre_acc, pre_lo, pre_hi, chance = _score(Xs_tr, y_tr, Xs_te, y_te, seed)
    rows = [{
        "eval_mode": eval_name, "direction": "(none)", "stage": "pre",
        "accuracy": pre_acc, "acc_ci_lo": pre_lo, "acc_ci_hi": pre_hi,
        "majority_chance": chance, "var_removed": 0.0,
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
    }]
    detail = {"pre": rows[0], "post": {}}

    if single_protocol_fit:
        print("      [warn] only one protocol class in train+test union; "
              "no protocol direction to remove for this eval.")
        return rows, detail

    for kind in DIRECTION_ESTIMATORS:
        try:
            d = _direction(kind, Xs_fit, protocol_fit, train_cell_fit)
        except Exception as e:
            print(f"      [warn] direction '{kind}' failed: {e}")
            continue
        var_removed = _variance_along(Xs_fit, d)
        Xs_tr_res = CL.project_out(Xs_tr, d)
        Xs_te_res = CL.project_out(Xs_te, d)
        acc, lo, hi, _ = _score(Xs_tr_res, y_tr, Xs_te_res, y_te, seed)
        row = {
            "eval_mode": eval_name, "direction": kind, "stage": "post",
            "accuracy": acc, "acc_ci_lo": lo, "acc_ci_hi": hi,
            "majority_chance": chance, "var_removed": var_removed,
            "drop_vs_pre": pre_acc - acc,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        }
        rows.append(row)
        detail["post"][kind] = row
        flag = " <== HEADLINE" if kind == HEADLINE else ""
        print(f"      {kind:10} post acc={acc:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  (pre {pre_acc:.3f}, "
              f"drop {pre_acc-acc:+.3f}, var_removed {var_removed:.3f}){flag}")
    return rows, detail


def main():
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    peak = CL.load_peak()
    L = peak["peak_layer"]
    print("=== C.3 orthogonalization (decisive test) ===")
    print(f"peak layer = {L}; directions = {DIRECTION_ESTIMATORS} "
          f"(headline: {HEADLINE})\n")

    labels_df = D.load_labels()
    response_ids = D.load_response_ids()
    y = CL.behavior_labels(response_ids, labels_df)
    protocol = _protocol_label(response_ids, labels_df)

    # in-distribution split (C.1's persisted random split)
    split = S.make_split(response_ids, labels_df)
    id_tr, id_te = S.split_masks(split, response_ids)

    # transfer split (C.2's cell split)
    tr_cell = CL.cell_mask(response_ids, labels_df, CL.TRAIN_CELL)
    te_cell = np.zeros(len(y), bool)
    for c in CL.TEST_CELLS:
        te_cell |= CL.cell_mask(response_ids, labels_df, c)

    all_rows, full = [], {"peak_layer": L, "directions": DIRECTION_ESTIMATORS,
                          "headline_direction": HEADLINE,
                          "limitation": ("linear removal of ONE combined "
                                         "protocol direction; nonlinear "
                                         "residual tested in D.2"),
                          "by_pooling": {}}

    for pooling in C.POOLINGS:
        print(f"----- pooling = {pooling} -----")
        X = D.load_layer_matrix(pooling, L)
        if X.shape[0] != len(response_ids):
            raise ValueError(f"row mismatch for {pooling} layer {L}")

        pooling_detail = {}
        for eval_name, (tm, em) in (
                ("in_distribution", (id_tr, id_te)),
                ("transfer", (tr_cell, te_cell))):
            if len(np.unique(y[tm])) < 2:
                print(f"  [{eval_name}] skip: train side <2 classes")
                continue
            if em.sum() == 0:
                print(f"  [{eval_name}] skip: empty test side")
                continue
            print(f"  [{eval_name}] train n={int(tm.sum())} "
                  f"test n={int(em.sum())}")
            rows, detail = _run_one_eval(eval_name, X, y, protocol,
                                         tm, em, C.SEED)
            for r in rows:
                r["pooling"] = pooling
                r["layer"] = L
            all_rows.extend(rows)
            pooling_detail[eval_name] = detail
        full["by_pooling"][pooling] = pooling_detail
        print()

    pd.DataFrame(all_rows).to_csv(C.OUT_DIR / "c3_orthogonalize.csv",
                                  index=False)
    with open(C.OUT_DIR / "c3_orthogonalize.json", "w") as f:
        json.dump(full, f, indent=2)
    print(f"Wrote {C.OUT_DIR/'c3_orthogonalize.csv'} and c3_orthogonalize.json")
    _interpret(full)


def _interpret(full):
    print("\n=== INTERPRETATION (headline: transfer + logistic direction) ===")
    for pooling, evals in full["by_pooling"].items():
        det = evals.get("transfer") or evals.get("in_distribution")
        if not det:
            continue
        pre = det["pre"]
        post = det["post"].get(HEADLINE)
        if not post:
            continue
        survives = post["acc_ci_lo"] > post["majority_chance"]
        verdict = ("SURVIVES -> evidence for a deception representation not "
                   "reducible to the protocol direction"
                   if survives else
                   "VANISHES toward majority -> the behavior signal was the "
                   "protocol signal in costume")
        mode = "transfer" if "transfer" in evals else "in_distribution"
        print(f"  [{pooling}/{mode}] pre={pre['accuracy']:.3f} -> "
              f"post={post['accuracy']:.3f} "
              f"[{post['acc_ci_lo']:.3f}, {post['acc_ci_hi']:.3f}] "
              f"(majority {post['majority_chance']:.3f}) -> {verdict}")
    print("  Scope: linear removal of one combined direction; D.2 checks for a "
          "surviving nonlinear confound.")


if __name__ == "__main__":
    main()
