"""C.2 confound-held-fixed transfer: train on one cell, test generalization to others."""
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
import cells as CL


def _within_cell_reference(X, y, train_mask, seed):
    """A within-distribution baseline: split the TRAIN cell itself into
    train/test and score there. This is the ceiling transfer is compared
    against — if even within-cell accuracy is at chance, the probe never
    learned anything and a transfer collapse is uninformative."""
    idx = np.where(train_mask)[0]
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_te = max(1, int(round(C.TEST_SIZE * len(idx))))
    te_idx, tr_idx = idx[:n_te], idx[n_te:]
    tr = np.zeros(len(y), bool); tr[tr_idx] = True
    te = np.zeros(len(y), bool); te[te_idx] = True
    # guard: need both classes in the within-cell train split
    if len(np.unique(y[tr])) < 2:
        return None
    return CL.fit_eval_behavior(X, y, tr, te, seed)


def main():
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    peak = CL.load_peak()
    L = peak["peak_layer"]
    print(f"=== C.2 confound-held-fixed transfer ===")
    print(f"peak layer = {L} (selected by C.1 on {peak['pooling']})\n")

    labels_df = D.load_labels()
    response_ids = D.load_response_ids()
    y = CL.behavior_labels(response_ids, labels_df)

    rows, full = [], {"peak_layer": L, "peak_meta": peak,
                      "train_cell": CL.TRAIN_CELL, "test_cells": CL.TEST_CELLS,
                      "caveat": ("transfer crosses strategy AND stake "
                                 "simultaneously; collinear, so a collapse "
                                 "is not attributable to one factor alone"),
                      "by_pooling": {}}

    train_mask = CL.cell_mask(response_ids, labels_df, CL.TRAIN_CELL)
    test_masks = [(f'{c["strategy"]}x{c["stake_structure"]}',
                   CL.cell_mask(response_ids, labels_df, c))
                  for c in CL.TEST_CELLS]
    pooled_test = np.zeros(len(y), bool)
    for _, m in test_masks:
        pooled_test |= m

    for pooling in C.POOLINGS:
        print(f"----- pooling = {pooling} -----")
        X = D.load_layer_matrix(pooling, L)
        if X.shape[0] != len(response_ids):
            raise ValueError(
                f"{C.act_path(pooling, L)} rows={X.shape[0]} != "
                f"response_ids={len(response_ids)}")

        print("  census:")
        census = [CL.describe_mask("TRAIN role_basedxprosocial",
                                   train_mask, y)]
        for name, m in test_masks:
            census.append(CL.describe_mask(f"TEST  {name}", m, y))
        census.append(CL.describe_mask("TEST  pooled self_serving",
                                       pooled_test, y))

        if len(np.unique(y[train_mask])) < 2:
            print("  [skip] TRAIN cell has <2 behavior classes — cannot "
                  "train a probe. (Expected if the model refused to produce "
                  "one class in this cell.)")
            full["by_pooling"][pooling] = {"census": census,
                                           "skipped": True}
            continue

        within = _within_cell_reference(X, y, train_mask, C.SEED)
        if within:
            print(f"  within-cell reference acc = {within['accuracy']:.3f} "
                  f"[{within['acc_ci_lo']:.3f}, {within['acc_ci_hi']:.3f}] "
                  "(ceiling for transfer)")

        conds = list(test_masks) + [("pooled_self_serving", pooled_test)]
        pooling_rows = []
        for cond_name, test_mask in conds:
            if test_mask.sum() == 0:
                print(f"  [skip] {cond_name}: empty")
                continue
            res = CL.fit_eval_behavior(X, y, train_mask, test_mask, C.SEED)
            chance = _majority_chance(y[test_mask])
            row = {
                "pooling": pooling, "layer": L, "test_condition": cond_name,
                "n_train": res["n_train"], "n_test": res["n_test"],
                "accuracy": res["accuracy"],
                "acc_ci_lo": res["acc_ci_lo"], "acc_ci_hi": res["acc_ci_hi"],
                "majority_chance": chance,
                "within_cell_ref": within["accuracy"] if within else None,
            }
            for cls, a in res["per_class_acc"].items():
                row[f"acc[{cls}]"] = a
            rows.append(row)
            pooling_rows.append({**row, "per_class_ci": res["per_class_ci"]})
            print(f"  {cond_name:24} acc={res['accuracy']:.3f} "
                  f"[{res['acc_ci_lo']:.3f}, {res['acc_ci_hi']:.3f}] "
                  f"(majority={chance:.3f}, n={res['n_test']})")

        full["by_pooling"][pooling] = {
            "census": census,
            "within_cell_reference": within,
            "transfer": pooling_rows,
        }
        print()

    pd.DataFrame(rows).to_csv(C.OUT_DIR / "c2_transfer.csv", index=False)
    with open(C.OUT_DIR / "c2_transfer.json", "w") as f:
        json.dump(full, f, indent=2)
    print(f"Wrote {C.OUT_DIR/'c2_transfer.csv'} and c2_transfer.json")
    _interpret(rows)


def _majority_chance(y_sub: np.ndarray) -> float:
    if len(y_sub) == 0:
        return float("nan")
    _, counts = np.unique(y_sub, return_counts=True)
    return float(counts.max() / counts.sum())


def _interpret(rows):
    print("\n=== INTERPRETATION ===")
    if not rows:
        print("  no transfer rows produced.")
        return
    for r in rows:
        if r["test_condition"] not in ("pooled_self_serving",):
            continue
        gap = r["accuracy"] - r["majority_chance"]
        verdict = ("TRANSFERS (probe carries signal across the boundary)"
                   if r["acc_ci_lo"] > r["majority_chance"]
                   else "COLLAPSES toward majority-class (consistent with the "
                        "probe riding the strategy/stake confound)")
        print(f"  [{r['pooling']}] pooled self_serving: "
              f"acc={r['accuracy']:.3f} vs majority={r['majority_chance']:.3f} "
              f"(gap {gap:+.3f}) -> {verdict}")
    print("  Reminder: a collapse here crosses strategy AND stake at once. "
          "C.3 isolates whether removing the protocol direction alone kills "
          "the probe.")


if __name__ == "__main__":
    main()
