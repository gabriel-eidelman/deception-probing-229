"""
split.py — make ONE stratified train/test split, persisted to disk so C.2,
C.3, and D.3 reuse the exact same held-out test set.

Stratification is JOINT over (behavior, strategy, stake_structure) so that rare
factorial cells are not lost from either side.  When a joint cell is too small
for sklearn's stratified split (a class with a single member can't be split),
we merge singleton cells into a pooled "rare" stratum and split that pooled
group with the same ratio — every example still lands in exactly one of train
or test, and no example is dropped.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C


def _joint_strata(labels_df: pd.DataFrame, response_ids: np.ndarray) -> np.ndarray:
    sub = labels_df.loc[response_ids]
    return (sub["behavior"].astype(str) + "|" +
            sub["strategy"].astype(str) + "|" +
            sub["stake_structure"].astype(str)).to_numpy()


def make_split(response_ids: np.ndarray, labels_df: pd.DataFrame) -> dict:
    """Build (or reuse) the persisted split. Returns
    {'seed', 'test_size', 'train_ids', 'test_ids'} with python int lists."""
    if C.SPLIT_PATH.exists():
        with open(C.SPLIT_PATH) as f:
            split = json.load(f)
        print(f"  reusing existing split at {C.SPLIT_PATH} "
              f"(seed={split['seed']}, "
              f"train={len(split['train_ids'])}, test={len(split['test_ids'])})")
        _check_coverage(split, response_ids)
        return split

    rng = np.random.RandomState(C.SEED)
    strata = _joint_strata(labels_df, response_ids)

    # Pool singleton strata so stratification is feasible.
    counts = pd.Series(strata).value_counts()
    rare = set(counts[counts < 2].index)
    strat_for_split = np.array(
        ["__rare__" if s in rare else s for s in strata]
    )
    if rare:
        print(f"  [info] {len(rare)} singleton joint-cells pooled into "
              f"'__rare__' for stratification feasibility.")

    train_ids, test_ids = train_test_split(
        response_ids,
        test_size=C.TEST_SIZE,
        random_state=rng,
        stratify=strat_for_split,
    )

    split = {
        "seed": C.SEED,
        "test_size": C.TEST_SIZE,
        "train_ids": sorted(int(i) for i in train_ids),
        "test_ids": sorted(int(i) for i in test_ids),
    }
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)
    print(f"  wrote split to {C.SPLIT_PATH}  "
          f"(seed={C.SEED}, train={len(train_ids)}, test={len(test_ids)})")
    _check_coverage(split, response_ids)
    return split


def _check_coverage(split: dict, response_ids: np.ndarray) -> None:
    ids = set(int(i) for i in response_ids)
    tr, te = set(split["train_ids"]), set(split["test_ids"])
    assert tr.isdisjoint(te), "train/test overlap!"
    covered = tr | te
    if covered != ids:
        missing = ids - covered
        extra = covered - ids
        raise ValueError(
            f"Split does not match current data. missing={len(missing)} "
            f"extra={len(extra)}. Delete {C.SPLIT_PATH} to regenerate if the "
            "dataset changed."
        )


def split_masks(split: dict, response_ids: np.ndarray):
    """Boolean masks over the activation row order for train / test."""
    tr = set(split["train_ids"])
    te = set(split["test_ids"])
    train_mask = np.array([rid in tr for rid in response_ids])
    test_mask = np.array([rid in te for rid in response_ids])
    return train_mask, test_mask
