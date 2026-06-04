"""Central configuration: paths, seed, and split location shared across all phases."""
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Paths.  Adjust ACTIVATIONS_DIR / SCORED_CSV to wherever Phase B wrote.
# ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PHASE_B_DIR = ROOT.parent.parent / "outputs"
SCORED_CSV = PHASE_B_DIR / "scored_responses.csv"        # frozen grader labels
ACTIVATIONS_DIR = PHASE_B_DIR / "activations"            # per-layer .npy files

OUT_DIR = ROOT / "outputs"
SPLIT_PATH = OUT_DIR / "split.json"                      # persisted train/test
RESULTS_CSV = OUT_DIR / "results.csv"
PEAK_LAYER_PATH = OUT_DIR / "peak_layer.json"

# ──────────────────────────────────────────────────────────────────────
# Reproducibility.
# ──────────────────────────────────────────────────────────────────────
SEED = 20260603

# ──────────────────────────────────────────────────────────────────────
# Activation file naming.  Phase B is expected to save, per pooling method
# and per layer, a float matrix of shape [n_examples, d_model] plus a row
# index that maps each row to a response_id in scored_responses.csv.
#
#   {ACTIVATIONS_DIR}/{pooling}_layer{L}.npy        # [N, d_model]
#   {ACTIVATIONS_DIR}/response_ids.npy              # [N]  (int response_id)
#
# If Phase B used a different scheme, edit act_path() / load_response_ids()
# in data_loading.py — that's the only coupling point.
# ──────────────────────────────────────────────────────────────────────
POOLINGS = ["last_token", "mean_pooled"]


def act_path(pooling: str, layer: int) -> Path:
    return ACTIVATIONS_DIR / f"{pooling}_layer{layer}.npy"


RESPONSE_IDS_PATH = ACTIVATIONS_DIR / "response_ids.npy"

# Set explicitly if you know the layer count; otherwise it's auto-detected
# from the files present in ACTIVATIONS_DIR.
LAYERS = None  # e.g. list(range(0, 80, 2)); None => auto-detect

# ──────────────────────────────────────────────────────────────────────
# Label extraction from scored_responses.csv.
# ──────────────────────────────────────────────────────────────────────
# behavior: frozen grader label (NEVER expected_behavior / generation intent)
BEHAVIOR_COL = "label"                 # values: "honest" / "deceptive"
BEHAVIOR_CLASSES = ["honest", "deceptive"]

# strategy: elicitation strategy
STRATEGY_COL = "elicitation_strategy"  # role_based / instructed / instrumental
STRATEGY_CLASSES = ["role_based", "instructed", "instrumental"]

# stake_structure: collapse raw "<prefix>_<...>" values to the 3 design classes.
# Edit this map if your raw stake_structure values differ.
STAKE_COL = "stake_structure"
STAKE_CLASSES = ["prosocial", "self_serving"]


def collapse_stake(raw: str) -> str:
    r = (raw or "").lower()
    if r.startswith("prosocial"):
        return "prosocial"
    if r.startswith("self_serving") or r.startswith("self-serving"):
        return "self_serving"
    if r.startswith("safety"):
        return "safety"
    # Fall back to the raw value so a mismatch is loud, not silently mislabeled.
    return raw


LABELS = {
    "behavior": {"col": BEHAVIOR_COL, "classes": BEHAVIOR_CLASSES,
                 "multinomial": False},
    "strategy": {"col": STRATEGY_COL, "classes": STRATEGY_CLASSES,
                 "multinomial": True},
    "stake_structure": {"col": STAKE_COL, "classes": STAKE_CLASSES,
                        "multinomial": True},
}

# ──────────────────────────────────────────────────────────────────────
# Probe / CV hyperparameters.
# ──────────────────────────────────────────────────────────────────────
TEST_SIZE = 0.20
N_CV_FOLDS = 5
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]   # inverse L2 strength
MAX_ITER = 2000
N_BOOTSTRAP = 1000
CI_ALPHA = 0.05   # 95% CI
