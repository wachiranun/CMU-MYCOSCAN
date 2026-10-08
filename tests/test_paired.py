import json

import pytest

from mycoscan.cli import main
from mycoscan.paired import PairedConfig, compare_runs

FOLDS = [f"fold{k}" for k in range(5)]


def _run(root, name, f1, weights="imagenet", arch="resnet18", seeds=(0,), membership=None, held_out="h0"):
    """A run directory as run_training leaves it: config.json and summary.json, with one isolate macro-F1
    per (seed, fold) taken from `f1` (one list per seed)."""
    run = root / name
    run.mkdir(parents=True)
    membership = membership or {f: f"m-{f}" for f in FOLDS}
    (run / "config.json").write_text(json.dumps({"run_name": name, "arch": arch, "weights": weights}))
    values = [{"seed": s, "fold": f, "value": v} for s, row in zip(seeds, f1) for f, v in zip(FOLDS, row)]
    (run / "summary.json").write_text(json.dumps({
        "run_name": name, "isolate_macro_f1": {"values": values},
        "provenance": {"seeds": list(seeds), "splits_sha256": "s1", "held_out": {"sha256": held_out},
                       "folds": {"names": FOLDS, "per_fold_sha256": membership}}}))
    return run


DIRECT = [0.80, 0.82, 0.78, 0.85, 0.75]


def test_known_per_fold_differences_give_the_mean_ci_and_superiority_verdict(tmp_path):
    gains = [0.03, 0.04, 0.05, 0.06, 0.07]
    seq = _run(tmp_path, "seq", [[d + g for d, g in zip(DIRECT, gains)]], weights=str(tmp_path / "stage1.pt"))
    direct = _run(tmp_path, "direct", [DIRECT])
    (result,) = compare_runs([seq, direct], PairedConfig(bootstrap=500))
    assert (result["sequential"], result["direct"], result["n_pairs"]) == ("seq", "direct", 5)
    assert result["mean_difference"] == pytest.approx(0.05)
    lo, hi = result["ci"]
    assert 0.03 <= lo < 0.05 < hi <= 0.07
    assert result["wilcoxon_p"] == pytest.approx(2 / 32)  # all five differences positive, exact test
    assert result["superior"] and not result["negative_transfer"]
    assert result["recommendation"] == "use sequential"


def test_a_gain_below_two_points_is_not_superior_even_when_the_ci_excludes_zero(tmp_path):
    seq = _run(tmp_path, "seq", [[d + 0.01 + 0.001 * k for k, d in enumerate(DIRECT)]], weights=str(tmp_path / "s1.pt"))
    (result,) = compare_runs([seq, _run(tmp_path, "direct", [DIRECT])], PairedConfig(bootstrap=500))
    assert result["ci"][0] > 0 and not result["superior"]
    assert result["thresholds"] == {"min_gain": 0.02, "ci_level": 0.95, "require_ci_excludes_zero": True}


def test_a_losing_sequential_run_is_named_negative_transfer_and_direct_is_recommended(tmp_path, capsys):
    seq = _run(tmp_path, "seq", [[d - 0.04 for d in DIRECT]], weights=str(tmp_path / "s1.pt"))
    direct = _run(tmp_path, "direct", [DIRECT])
    main(["paired", str(seq), str(direct), "--set", "bootstrap=200", "--out", str(tmp_path / "paired.json")])
    out = capsys.readouterr().out
    assert "negative transfer" in out.lower()
    assert "use direct" in out
    saved = json.loads((tmp_path / "paired.json").read_text())
    assert saved["config"]["min_gain"] == 0.02 and saved["config"]["bootstrap"] == 200
    assert saved["pairs"][0]["negative_transfer"] is True


def test_runs_with_different_fold_membership_are_refused_naming_the_first_mismatched_fold(tmp_path):
    moved = {f: f"m-{f}" for f in FOLDS} | {"fold2": "other", "fold4": "other"}
    seq = _run(tmp_path, "seq", [DIRECT], weights=str(tmp_path / "s1.pt"), membership=moved)
    with pytest.raises(ValueError, match="fold2"):
        compare_runs([seq, _run(tmp_path, "direct", [DIRECT])], PairedConfig())


def test_runs_with_different_seeds_or_test_isolates_are_refused(tmp_path):
    seq = _run(tmp_path, "seq", [DIRECT], weights=str(tmp_path / "s1.pt"), seeds=(1,))
    with pytest.raises(ValueError, match="seeds"):
        compare_runs([seq, _run(tmp_path, "direct", [DIRECT])], PairedConfig())
    seq = _run(tmp_path, "seq2", [DIRECT], weights=str(tmp_path / "s1.pt"), held_out="h1")
    with pytest.raises(ValueError, match="test isolates"):
        compare_runs([seq, _run(tmp_path, "direct2", [DIRECT])], PairedConfig())


def test_paired_comparison_reads_two_multi_seed_runs_without_reshaping(synthetic_manifest, tmp_path):
    from mycoscan.config import Config
    from mycoscan.models import build_model, save_checkpoint
    from mycoscan.pipeline import run_training

    stage1 = tmp_path / "stage1.pt"
    save_checkpoint(stage1, build_model("resnet18", 5, "none", 32), "resnet18", list("abcde"), 32, True, {})
    base = dict(manifest=str(synthetic_manifest), output_dir=str(tmp_path / "runs"), modality="microscopic",
                source="cmu", arch="resnet18", image_size=32, epochs=1, batch_size=8, bootstrap=0, device="cpu",
                split="kfold", n_folds=2, seeds=(1, 2), fit_final=False, tau_rule="none")
    direct = run_training(Config(run_name="direct", weights="none", **base))
    seq = run_training(Config(run_name="seq", weights=str(stage1), **base))
    (result,) = compare_runs([direct, seq], PairedConfig(bootstrap=100))
    assert (result["sequential"], result["direct"], result["n_pairs"]) == ("seq", "direct", 4)
    assert sorted((d["seed"], d["fold"]) for d in result["differences"]) == [
        (1, "fold0"), (1, "fold1"), (2, "fold0"), (2, "fold1")]
