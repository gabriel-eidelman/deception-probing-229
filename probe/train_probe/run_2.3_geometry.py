"""D.3 geometry: cosine similarity between behavior probes trained on disjoint strategy cells."""
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

STRATEGY_A = "role_based"
STRATEGY_B = "instrumental"
N_PERM = 200


def _strategy_mask(response_ids, labels_df, strategy):
    strat = D.aligned_labels(response_ids, labels_df, "strategy")
    return strat == strategy


def _probe_vectors(pipe):
    """Return (w_std, w_raw) for the behavior probe.
    w_std lives in standardized space (the LogReg coef); w_raw maps it back to
    raw activation space by dividing out the scaler's per-feature std, which
    makes the two probes' vectors comparable despite different scalers."""
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    w_std = clf.coef_.ravel().astype(float)
    sigma = scaler.scale_.astype(float)
    sigma = np.where(sigma == 0, 1.0, sigma)
    w_raw = w_std / sigma
    return w_std, w_raw


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _fit_strategy_probe(X, y, mask, seed):
    if len(np.unique(y[mask])) < 2:
        return None
    pipe, _ = CL.fitted_behavior_probe(X, y, mask, seed)
    return pipe


def main():
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    peak = CL.load_peak()
    L = peak["peak_layer"]
    print("=== D.3 geometry: one construct or two? ===")
    print(f"peak layer = {L}; comparing {STRATEGY_A} vs {STRATEGY_B} "
          "behavior probes\n")

    labels_df = D.load_labels()
    response_ids = D.load_response_ids()
    y = CL.behavior_labels(response_ids, labels_df)
    mask_A = _strategy_mask(response_ids, labels_df, STRATEGY_A)
    mask_B = _strategy_mask(response_ids, labels_df, STRATEGY_B)

    rows, full = [], {"peak_layer": L, "strategy_a": STRATEGY_A,
                      "strategy_b": STRATEGY_B, "n_perm": N_PERM,
                      "by_pooling": {}}

    for pooling in C.POOLINGS:
        print(f"----- pooling = {pooling} -----")
        CL.describe_mask(f"{STRATEGY_A} cells", mask_A, y)
        CL.describe_mask(f"{STRATEGY_B} cells", mask_B, y)
        X = D.load_layer_matrix(pooling, L)

        pipe_A = _fit_strategy_probe(X, y, mask_A, C.SEED)
        pipe_B = _fit_strategy_probe(X, y, mask_B, C.SEED)
        if pipe_A is None or pipe_B is None:
            which = STRATEGY_A if pipe_A is None else STRATEGY_B
            print(f"  [skip] {which} cells lack both behavior classes — "
                  "cannot fit a probe. (Plausible given model resistance in "
                  "some cells; report as a census finding instead.)")
            full["by_pooling"][pooling] = {"skipped": which}
            continue

        wA_std, wA_raw = _probe_vectors(pipe_A)
        wB_std, wB_raw = _probe_vectors(pipe_B)
        cos_std = _cos(wA_std, wB_std)
        cos_raw = _cos(wA_raw, wB_raw)

        # permutation null in raw space (the headline comparison)
        rng = np.random.RandomState(C.SEED)
        null = []
        for _ in range(N_PERM):
            yp = y.copy()
            for m in (mask_A, mask_B):
                idx = np.where(m)[0]
                yp[idx] = rng.permutation(yp[idx])
            pA = _fit_strategy_probe(X, yp, mask_A, C.SEED)
            pB = _fit_strategy_probe(X, yp, mask_B, C.SEED)
            if pA is None or pB is None:
                continue
            _, a = _probe_vectors(pA)
            _, b = _probe_vectors(pB)
            null.append(abs(_cos(a, b)))
        null = np.array(null) if null else np.array([np.nan])
        null_lo, null_hi = (float(np.nanpercentile(null, 2.5)),
                            float(np.nanpercentile(null, 97.5)))
        above_null = bool(abs(cos_raw) > null_hi)

        row = {
            "pooling": pooling, "layer": L,
            "cosine_raw": cos_raw, "cosine_std": cos_std,
            "null_abs_cos_lo": null_lo, "null_abs_cos_hi": null_hi,
            "above_null": above_null,
            "n_A": int(mask_A.sum()), "n_B": int(mask_B.sum()),
        }
        rows.append(row)
        full["by_pooling"][pooling] = row
        print(f"  cosine (raw space)  = {cos_raw:+.3f}   <== headline")
        print(f"  cosine (std space)  = {cos_std:+.3f}")
        print(f"  permutation null |cos| 95% band = "
              f"[{null_lo:.3f}, {null_hi:.3f}]")
        print(f"  -> {'ALIGNED beyond chance' if above_null else 'within chance band'}\n")

    pd.DataFrame(rows).to_csv(C.OUT_DIR / "d3_geometry.csv", index=False)
    with open(C.OUT_DIR / "d3_geometry.json", "w") as f:
        json.dump(full, f, indent=2)
    print(f"Wrote {C.OUT_DIR/'d3_geometry.csv'} and d3_geometry.json")
    _interpret(rows)


def _interpret(rows):
    print("\n=== INTERPRETATION ===")
    if not rows:
        print("  no geometry rows — at least one strategy cell lacked both "
              "behavior classes. That asymmetry is itself a census finding "
              "(E.2): honesty training binds where the model won't produce "
              "deceptive responses.")
        return
    for r in rows:
        if not r["above_null"]:
            v = ("cosine within the chance band -> the two strategies' "
                 "behavior probes are not meaningfully aligned at this sample "
                 "size; consistent with TWO strategy-specific signals")
        elif r["cosine_raw"] > 0.5:
            v = ("strongly aligned beyond chance -> ONE behavior construct, "
                 "consistent across elicitation strategy")
        else:
            v = ("aligned beyond chance but weakly -> partial sharing; "
                 "report the magnitude honestly, not just the sign")
        print(f"  [{r['pooling']}] cos_raw={r['cosine_raw']:+.3f} "
              f"(null<= {r['null_abs_cos_hi']:.3f}) -> {v}")


if __name__ == "__main__":
    main()
