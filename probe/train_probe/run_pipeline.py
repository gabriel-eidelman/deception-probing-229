"""C.1 end-to-end decodability-by-factor pipeline: split, probe training, and layer plots."""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd

# quiet excess sklearn deprecation warnings them so the run summary stays readable.
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

import config as C
import data_loading as D
import split as S
import probes as P
import plotting as PL


def main():
    np.random.seed(C.SEED)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== C.1 decodability-by-factor ===\nSEED = {C.SEED}\n")

    # 1. Labels + activation row index.
    print("Loading labels...")
    labels_df = D.load_labels()
    response_ids = D.load_response_ids()
    print(f"\n{len(response_ids)} activation rows; "
          f"{len(labels_df)} labeled responses.")

    # 2. One stratified split, persisted.
    print("\nBuilding / loading the persisted split...")
    split = S.make_split(response_ids, labels_df)
    train_mask, test_mask = S.split_masks(split, response_ids)

    # 3. Detect layers.
    layers = D.detect_layers()
    print(f"\nLayers detected: {layers[0]}..{layers[-1]} ({len(layers)} total)")

    # 4. Fit and evaluate every pooling x label x layer.
    results = []
    for pooling in C.POOLINGS:
        print(f"\n----- pooling = {pooling} -----")
        for layer in layers:
            X = D.load_layer_matrix(pooling, layer)
            if X.shape[0] != len(response_ids):
                raise ValueError(
                    f"{C.act_path(pooling, layer)} has {X.shape[0]} rows but "
                    f"response_ids has {len(response_ids)}.")
            X_tr, X_te = X[train_mask], X[test_mask]
            for label_name, spec in C.LABELS.items():
                y = D.aligned_labels(response_ids, labels_df, label_name)
                y_tr, y_te = y[train_mask], y[test_mask]
                res = P.evaluate_probe(
                    label_name, pooling, layer,
                    X_tr, y_tr, X_te, y_te,
                    spec["classes"], spec["multinomial"], C.SEED,
                )
                results.append(res)
            print(f"  layer {layer:>3}: " + "  ".join(
                f"{r['label'][:4]}={r['accuracy']:.3f}"
                for r in results[-len(C.LABELS):]))

    # 5. Results table.
    _write_results_table(results)

    # 6. Peak layer for the behavior probe (selected on mean_pooled if present,
    #    else last_token — surfaced explicitly for C.2/C.3/D.3).
    peak = _select_peak_layer(results)
    with open(C.PEAK_LAYER_PATH, "w") as f:
        json.dump(peak, f, indent=2)
    print(f"\nPEAK BEHAVIOR LAYER = {peak['peak_layer']} "
          f"(pooling={peak['pooling']}, acc={peak['accuracy']:.3f}, "
          f"95% CI [{peak['acc_ci_lo']:.3f}, {peak['acc_ci_hi']:.3f}])")
    print(f"  written to {C.PEAK_LAYER_PATH}")

    # 7. Plots.
    for pooling in C.POOLINGS:
        p = PL.plot_accuracy_vs_layer(results, pooling, C.OUT_DIR)
        if p:
            print(f"  wrote {p}")

    # 8. Interpretation.
    _print_interpretation(results, peak)


def _write_results_table(results):
    rows = []
    for r in results:
        row = {k: r[k] for k in ("label", "pooling", "layer", "best_C",
                                 "n_train", "n_test",
                                 "accuracy", "acc_ci_lo", "acc_ci_hi")}
        for cls, acc in r["per_class_acc"].items():
            row[f"acc[{cls}]"] = acc
            lo, hi = r["per_class_ci"][cls]
            row[f"ci_lo[{cls}]"] = lo
            row[f"ci_hi[{cls}]"] = hi
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_CSV, index=False)
    print(f"\nWrote results table -> {C.RESULTS_CSV} ({len(df)} rows)")


def _select_peak_layer(results):
    pooling = "mean_pooled" if any(
        r["pooling"] == "mean_pooled" for r in results) else "last_token"
    beh = [r for r in results
           if r["label"] == "behavior" and r["pooling"] == pooling]
    best = max(beh, key=lambda r: r["accuracy"])
    return {
        "peak_layer": int(best["layer"]),
        "pooling": pooling,
        "accuracy": best["accuracy"],
        "acc_ci_lo": best["acc_ci_lo"],
        "acc_ci_hi": best["acc_ci_hi"],
        "note": "C.2 / C.3 / D.3 run the behavior probe at this layer.",
    }


def _print_interpretation(results, peak):
    print("\n=== INTERPRETATION: strategy/stake vs. behavior ===")
    pooling = peak["pooling"]
    L = peak["peak_layer"]

    def at(label):
        m = [r for r in results
             if r["label"] == label and r["pooling"] == pooling
             and r["layer"] == L]
        return m[0] if m else None

    b, s, k = at("behavior"), at("strategy"), at("stake_structure")
    print(f"At peak layer {L} ({pooling}):")
    for tag, r in (("behavior", b), ("strategy", s), ("stake_structure", k)):
        if r:
            print(f"  {tag:16}: acc={r['accuracy']:.3f} "
                  f"[{r['acc_ci_lo']:.3f}, {r['acc_ci_hi']:.3f}]")
    if b and s and k:
        print(
            "\nNote: behavior is confounded with the protocol (strategy/stake) "
            "by design. If strategy/stake decode as well as or better than "
            "behavior at this layer, an apparent 'deception' probe may be "
            "reading the protocol rather than the behavior itself. Compare the "
            "CIs — overlapping intervals mean the data can't separate the two."
        )


if __name__ == "__main__":
    main()
