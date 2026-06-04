"""Run the full probe pipeline in required order (census → C.1 → C.2 → C.3 → D.2 → D.3)."""
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
