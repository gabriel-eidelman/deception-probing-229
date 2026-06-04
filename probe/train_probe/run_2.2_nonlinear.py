"""D.2 nonlinear check: MLP probe before and after linear protocol direction removal."""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

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

# Small MLP — sample sizes here are tiny, so capacity is deliberately limited
# and regularization (alpha) deliberately strong, to avoid an MLP that simply
# memorizes the train cell. Reported with the same bootstrap CI as everything
# else; CIs will be wide (D.4).
MLP_KW = dict(hidden_layer_sizes=(64,), alpha=1.0, max_iter=2000,
              early_stopping=False)


def _protocol_label(response_ids, labels_df):
    strat = D.aligned_labels(response_ids, labels_df, "strategy")
    stake = D.aligned_labels(response_ids, labels_df, "stake_structure")
    return np.array([f"{s}|{k}" for s, k in zip(strat, stake)])


def _majority_chance(y_sub):
    if len(y_sub) == 0:
        return float("nan")
    _, counts = np.unique(y_sub, return_counts=True)
    return float(counts.max() / counts.sum())


def _eval(clf, Xs_tr, y_tr, Xs_te, y_te, seed):
    clf.fit(Xs_tr, y_tr)
    pred = clf.predict(Xs_te)
    acc = float((pred == y_te).mean())
    lo, hi = CL.accuracy_ci(y_te, pred, seed)
    return acc, lo, hi


def _run(eval_name, X, y, protocol, tm, em, seed):
    scaler = CL.safe_scaler(X[tm])
    Xs_tr, Xs_te = scaler.transform(X[tm]), scaler.transform(X[em])
    y_tr, y_te = y[tm], y[em]
    chance = _majority_chance(y_te)

    # protocol direction on the train+test union (train cell may be single-
    # protocol in transfer mode), in the same standardized space.
    fit_mask = tm | em
    Xs_fit = scaler.transform(X[fit_mask])
    protocol_fit = protocol[fit_mask]
    if len(set(protocol_fit)) < 2:
        print("    [warn] one protocol class in union; skipping post-removal.")
        Xs_tr_res, Xs_te_res = Xs_tr, Xs_te
    else:
        d = CL.protocol_direction_logistic(Xs_fit, protocol_fit, seed)
        Xs_tr_res, Xs_te_res = CL.project_out(Xs_tr, d), CL.project_out(Xs_te, d)

    rows = []
    for clf_name, make in (("linear", lambda: CL.make_logreg(multinomial=False)),
                           ("mlp", lambda: MLPClassifier(
                               random_state=seed, **MLP_KW))):
        for stage, (Xtr, Xte) in (("pre", (Xs_tr, Xs_te)),
                                  ("post", (Xs_tr_res, Xs_te_res))):
            acc, lo, hi = _eval(make(), Xtr, y_tr, Xte, y_te, seed)
            rows.append({
                "eval_mode": eval_name, "probe": clf_name, "stage": stage,
                "accuracy": acc, "acc_ci_lo": lo, "acc_ci_hi": hi,
                "majority_chance": chance,
                "n_train": int(tm.sum()), "n_test": int(em.sum()),
            })
            print(f"    {clf_name:6} {stage:4} acc={acc:.3f} "
                  f"[{lo:.3f}, {hi:.3f}] (majority {chance:.3f})")
    return rows


def main():
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    peak = CL.load_peak()
    L = peak["peak_layer"]
    print("=== D.2 nonlinear check (MLP vs linear, pre/post orthogonalization) ===")
    print(f"peak layer = {L}\n")

    labels_df = D.load_labels()
    response_ids = D.load_response_ids()
    y = CL.behavior_labels(response_ids, labels_df)
    protocol = _protocol_label(response_ids, labels_df)

    split = S.make_split(response_ids, labels_df)
    id_tr, id_te = S.split_masks(split, response_ids)
    tr_cell = CL.cell_mask(response_ids, labels_df, CL.TRAIN_CELL)
    te_cell = np.zeros(len(y), bool)
    for c in CL.TEST_CELLS:
        te_cell |= CL.cell_mask(response_ids, labels_df, c)

    all_rows, full = [], {"peak_layer": L, "mlp_kw": MLP_KW, "by_pooling": {}}
    for pooling in C.POOLINGS:
        print(f"----- pooling = {pooling} -----")
        X = D.load_layer_matrix(pooling, L)
        pooling_rows = {}
        for eval_name, (tm, em) in (("transfer", (tr_cell, te_cell)),
                                    ("in_distribution", (id_tr, id_te))):
            if len(np.unique(y[tm])) < 2 or em.sum() == 0:
                print(f"  [{eval_name}] skip")
                continue
            print(f"  [{eval_name}] train n={int(tm.sum())} test n={int(em.sum())}")
            rows = _run(eval_name, X, y, protocol, tm, em, C.SEED)
            for r in rows:
                r["pooling"] = pooling; r["layer"] = L
            all_rows.extend(rows)
            pooling_rows[eval_name] = rows
        full["by_pooling"][pooling] = pooling_rows
        print()

    pd.DataFrame(all_rows).to_csv(C.OUT_DIR / "d2_nonlinear.csv", index=False)
    with open(C.OUT_DIR / "d2_nonlinear.json", "w") as f:
        json.dump(full, f, indent=2)
    print(f"Wrote {C.OUT_DIR/'d2_nonlinear.csv'} and d2_nonlinear.json")
    _interpret(all_rows)


def _interpret(rows):
    print("\n=== INTERPRETATION ===")
    tr = [r for r in rows if r["eval_mode"] == "transfer"]
    for pooling in C.POOLINGS:
        sub = {(r["probe"], r["stage"]): r for r in tr if r["pooling"] == pooling}
        if ("mlp", "post") not in sub or ("linear", "post") not in sub:
            continue
        lin_post, mlp_post = sub[("linear", "post")], sub[("mlp", "post")]
        mlp_recovers = mlp_post["acc_ci_lo"] > mlp_post["majority_chance"]
        lin_gone = lin_post["acc_ci_lo"] <= lin_post["majority_chance"]
        if lin_gone and mlp_recovers:
            v = ("NONLINEAR residual SURVIVES linear removal — linear-only "
                 "caveat (D.1) matters")
        elif mlp_recovers:
            v = "both linear and nonlinear behavior survive"
        else:
            v = "no nonlinear residual — linear removal was sufficient"
        print(f"  [{pooling}/transfer] linear_post={lin_post['accuracy']:.3f}, "
              f"mlp_post={mlp_post['accuracy']:.3f} -> {v}")


if __name__ == "__main__":
    main()
