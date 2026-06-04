"""
run_all.py — run the full probe competition end to end, in the order the plan
requires (order matters: C.1 selects the peak layer everything else runs at).

    cd probe/train_probe
    python run_all.py

Sequence:
    report_census      (E.2 behavioral backbone; independent of probes)
    C.1 run_pipeline   (decodability-by-factor; writes peak_layer.json)
    C.2 run_c2         (confound-held-fixed transfer)
    C.3 run_c3         (orthogonalization — decisive test)
    D.2 run_d2         (nonlinear check)
    D.3 run_d3         (geometry)

D.4 (bootstrap CIs) is not a separate stage — every accuracy/transfer number in
every stage already carries a percentile bootstrap CI via probes.py /
cells.accuracy_ci. With these sample sizes the CIs are wide by design; report
them, don't lean on point estimates.

Each stage is imported and called in-process so a failure stops the chain with
a clear message rather than silently skipping downstream work.
"""
from __future__ import annotations
import sys
import traceback

STAGES = [
    ("census (E.2)", "report_census"),
    ("C.1 decodability-by-factor", "run_pipeline"),
    ("C.2 transfer", "run_c2_transfer"),
    ("C.3 orthogonalization", "run_c3_orthogonalize"),
    ("D.2 nonlinear check", "run_d2_nonlinear"),
    ("D.3 geometry", "run_d3_geometry"),
]


def main():
    failures = []
    for label, mod_name in STAGES:
        print("\n" + "=" * 72)
        print(f">>> {label}  ({mod_name}.py)")
        print("=" * 72)
        try:
            mod = __import__(mod_name)
            mod.main()
        except SystemExit:
            raise
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n!!! STAGE FAILED: {label}\n{tb}")
            failures.append((label, str(e)))
            # C.2/C.3/D.* all depend on C.1's peak_layer.json; if C.1 failed,
            # stop rather than cascade confusing errors.
            if mod_name == "run_pipeline":
                print("C.1 failed — peak_layer.json not written; "
                      "halting the chain.")
                break

    print("\n" + "=" * 72)
    if failures:
        print("COMPLETED WITH FAILURES:")
        for label, err in failures:
            print(f"  - {label}: {err}")
        sys.exit(1)
    print("All stages completed. Outputs in ./outputs/")


if __name__ == "__main__":
    main()
