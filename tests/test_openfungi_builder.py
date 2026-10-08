import json
from pathlib import Path

import pandas as pd
import pytest

from mycoscan.cli import main
from mycoscan.config import Config
from mycoscan.pipeline import run_training
from mycoscan.manifest import load_manifest
from mycoscan.openfungi import GroupingConfig, build_openfungi_manifest, load_grouping_config
from mycoscan.synthetic import make_openfungi_folders

# A tiny random-weight backbone from the same registry stands in for the DINO embedding.
STUB = dict(embed_arch="resnet18", embed_weights="none", embed_image_size=32, device="cpu")


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    return make_openfungi_folders(tmp_path_factory.mktemp("openfungi"))


def _build(folders, out_dir, **kw):
    manifest, summary = build_openfungi_manifest(folders, out_dir / "manifest.csv", GroupingConfig(**{**STUB, **kw}))
    rows = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    rows["plate"] = rows["image_path"].str.extract(r"((?:macro|micro)/.+)_shot\d\.jpg$")[0]
    return rows, summary


def test_planted_duplicates_share_a_group_and_distinct_plates_do_not(folders, tmp_path):
    rows, _ = _build(folders, tmp_path)
    assert rows["plate"].notna().all()
    assert (rows.groupby("plate")["group_id"].nunique() == 1).all()
    assert (rows.groupby("group_id")["plate"].nunique() == 1).all()
    assert rows["group_id"].nunique() == rows["plate"].nunique()


def test_thresholds_are_config_keys_and_hamming_zero_splits_the_planted_groups(folders, tmp_path):
    toml = tmp_path / "grouping.toml"
    toml.write_text("hamming_threshold = 8\ncosine_threshold = 0.15\n", encoding="utf-8")
    cfg = load_grouping_config(toml, ["hamming_threshold=0", 'embed_arch="resnet18"', 'embed_weights="none"',
                                      "embed_image_size=32", 'device="cpu"'])
    assert (cfg.hamming_threshold, cfg.cosine_threshold) == (0, 0.15)
    manifest, _ = build_openfungi_manifest(folders, tmp_path / "manifest.csv", cfg)
    rows = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    rows["plate"] = rows["image_path"].str.extract(r"((?:macro|micro)/.+)_shot\d\.jpg$")[0]
    planted = rows.groupby("plate").size()
    split = rows.groupby("plate")["group_id"].nunique()
    multi_shot = planted[planted > 1].index
    assert (split[multi_shot] > 1).mean() > 0.5
    with pytest.raises(ValueError, match="hamming"):
        load_grouping_config(None, ["hamming=3"])


def test_manifest_loads_and_leaves_mixed_out_unless_asked(folders, tmp_path):
    rows, _ = _build(folders, tmp_path / "default")
    df = load_manifest(tmp_path / "default" / "manifest.csv")
    assert len(df) == len(rows) and (df["source"] == "openfungi").all()
    assert set(df["modality"]) == {"colony", "microscopic"}
    assert "Mixed" not in set(df["species"])
    assert df["group"].tolist() == df["group_id"].tolist()
    assert {"sha256", "phash", "width", "height"} <= set(rows)
    assert (rows["width"] == "96").all()
    with_mixed, _ = _build(folders, tmp_path / "mixed", include_mixed=True)
    assert "Mixed" in set(with_mixed["species"])


def test_one_contact_sheet_per_group_with_more_than_one_image(folders, tmp_path):
    rows, _ = _build(folders, tmp_path)
    sizes = rows.groupby("group_id").size()
    sheets = {p.stem for p in (tmp_path / "contact_sheets").glob("*.png")}
    assert sheets == set(sizes[sizes > 1].index)


def test_summary_counts_images_per_class_and_flags_classes_under_twenty(folders, tmp_path):
    rows, summary = _build(folders, tmp_path, min_images=7)
    counts = {(c["modality"], c["species"]): c for c in summary["classes"]}
    flavi = counts[("colony", "Aspergillus_section_Flavi")]
    assert flavi["images"] == 7 and flavi["groups"] == 4  # 4 plates shot 1, 2, 3 and 1 times
    assert not flavi["under_powered"]
    assert GroupingConfig().min_images == 20
    _, default = _build(folders, tmp_path / "default")
    assert len(default["under_powered"]) == len(default["classes"]) == 8


def test_cli_builds_the_manifest_and_prints_the_summary(folders, tmp_path, capsys):
    main(["build-openfungi", "--root", str(folders), "--out", str(tmp_path / "m.csv"), "--set", "hamming_threshold=8",
          *[a for k, v in STUB.items() for a in ("--set", f"{k}={v!r}".replace("'", '"'))]])
    printed = capsys.readouterr().out
    assert "Aspergillus_section_Flavi" in printed and "under-powered" in printed
    assert load_manifest(tmp_path / "m.csv")["group_id"].str.startswith("OF_").all()


def test_select_classes_trains_and_scores_only_the_listed_subset(synthetic_manifest, tmp_path):
    subset = ("Aspergillus", "Penicillium", "Alternaria")
    run_dir = run_training(Config(run_name="subset", manifest=str(synthetic_manifest), output_dir=str(tmp_path),
                                  source="openfungi", modality="colony", arch="resnet18", weights="none",
                                  image_size=32, epochs=1, batch_size=8, bootstrap=0, device="cpu", split="holdout",
                                  select_classes=subset))
    assert set(pd.read_csv(run_dir / "predictions.csv")["species"]) <= set(subset)
    assert json.loads((run_dir / "metrics.json").read_text())["scored_classes"] == sorted(subset)
    with pytest.raises(ValueError, match="Fusarium_spp"):
        run_training(Config(run_name="bad", manifest=str(synthetic_manifest), output_dir=str(tmp_path),
                            source="openfungi", modality="colony", weights="none", select_classes=("Fusarium_spp",)))


@pytest.mark.network
def test_real_openfungi_folders_build_a_manifest_that_loads(tmp_path):
    root = Path(__file__).resolve().parents[1] / "openfungi"
    if not root.is_dir():
        pytest.skip("the OpenFungi folders are not in this checkout")
    manifest, summary = build_openfungi_manifest(root, tmp_path / "manifest.csv")
    df = load_manifest(manifest)
    assert "Mixed" not in set(df["species"])
    assert {"colony/Fusarium_spp", "colony/Rhizopus_spp"} <= set(summary["under_powered"])
