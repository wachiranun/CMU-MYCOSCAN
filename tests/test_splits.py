import numpy as np
import pandas as pd
import pytest

from mycoscan.splits import Fold, assert_no_group_leakage, make_folds


def _manifest_rows():
    rows = []
    for c in range(10):
        for i in range(3 if c % 2 == 0 else 2):
            iso = f"S{c}-I{i}"
            for device in ("microscope_camera", "smartphone"):
                for fov in range(4):
                    rows.append({"species": f"S{c}", "isolate_id": iso, "group": iso, "device": device, "fov": fov})
    return pd.DataFrame(rows)


def _val_isolates(df, fold):
    return set(df["isolate_id"].iloc[fold.val_idx])


def _train_isolates(df, fold):
    return set(df["isolate_id"].iloc[fold.train_idx])


@pytest.mark.parametrize("strategy", ["holdout", "kfold", "loio"])
def test_no_isolate_appears_on_both_sides(strategy):
    df = _manifest_rows()
    for fold in make_folds(df, strategy, n_folds=5, val_fraction=0.2, seed=1):
        assert _val_isolates(df, fold) & _train_isolates(df, fold) == set()
        assert sorted(np.concatenate([fold.train_idx, fold.val_idx]).tolist()) == list(range(len(df)))


def test_kfold_validates_every_isolate_once_in_equal_folds():
    df = _manifest_rows()
    folds = make_folds(df, "kfold", n_folds=5, seed=7)
    assert [len(_val_isolates(df, f)) for f in folds] == [5, 5, 5, 5, 5]
    seen = [iso for f in folds for iso in _val_isolates(df, f)]
    assert len(seen) == 25 and len(set(seen)) == 25


def test_kfold_spreads_each_class_across_folds():
    df = _manifest_rows()
    folds = make_folds(df, "kfold", n_folds=5, seed=3)
    for species, n_isolates in df.groupby("species")["isolate_id"].nunique().items():
        folds_holding_class = [f.name for f in folds if species in set(df["species"].iloc[f.val_idx])]
        assert len(folds_holding_class) == n_isolates


def test_holdout_is_one_fifth_of_isolates():
    df = _manifest_rows()
    [fold] = make_folds(df, "holdout", val_fraction=0.2, seed=0)
    assert len(_val_isolates(df, fold)) == 5
    assert len(_train_isolates(df, fold)) == 20


def test_loio_has_one_fold_per_isolate():
    df = _manifest_rows()
    folds = make_folds(df, "loio")
    assert len(folds) == 25
    assert all(len(_val_isolates(df, f)) == 1 for f in folds)


def test_folds_keep_a_group_together_even_when_isolate_ids_differ():
    rows = []
    for c in range(3):
        for g in range(4):
            for shot in range(2):
                rows.append({"species": f"S{c}", "isolate_id": f"img:S{c}-G{g}-{shot}", "group": f"S{c}-G{g}"})
    df = pd.DataFrame(rows)
    for fold in make_folds(df, "kfold", n_folds=2, seed=0):
        train_groups = set(df["group"].iloc[fold.train_idx])
        val_groups = set(df["group"].iloc[fold.val_idx])
        assert train_groups & val_groups == set()
        assert_no_group_leakage(df, fold)


def test_leakage_detector_rejects_shared_isolate():
    df = _manifest_rows()
    same_isolate = np.flatnonzero(df["group"] == "S0-I0")
    rest = np.flatnonzero(df["group"] != "S0-I0")
    leaky = Fold("leaky", np.concatenate([rest, same_isolate[:1]]), same_isolate[1:])
    with pytest.raises(AssertionError, match="S0-I0"):
        assert_no_group_leakage(df, leaky)
