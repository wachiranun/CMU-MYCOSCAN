import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mycoscan.manifest import load_manifest


def _write(tmp_path, rows):
    for r in rows:
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / r["image_path"])
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_cmu_rows_group_by_isolate(tmp_path):
    path = _write(tmp_path, [
        {"image_path": "a.png", "species": "S1", "isolate_id": "I1", "modality": "microscopic"},
        {"image_path": "b.png", "species": "S1", "isolate_id": "I1", "modality": "colony"},
        {"image_path": "c.png", "species": "S2", "isolate_id": "I2", "modality": "microscopic"},
    ])
    df = load_manifest(path)
    assert df["group"].tolist() == ["I1", "I1", "I2"]


def test_openfungi_rows_group_by_group_id(tmp_path):
    path = _write(tmp_path, [
        {"image_path": "a.png", "species": "Flavi", "isolate_id": "", "modality": "colony", "source": "openfungi", "group_id": "G7"},
        {"image_path": "b.png", "species": "Flavi", "isolate_id": "", "modality": "colony", "source": "openfungi", "group_id": "G7"},
        {"image_path": "c.png", "species": "Nigri", "isolate_id": "", "modality": "colony", "source": "openfungi", "group_id": "G9"},
    ])
    df = load_manifest(path)
    assert df["group"].tolist() == ["G7", "G7", "G9"]


def test_openfungi_row_without_group_id_is_rejected(tmp_path):
    path = _write(tmp_path, [
        {"image_path": "a.png", "species": "Flavi", "isolate_id": "", "modality": "colony", "source": "openfungi", "group_id": "G7"},
        {"image_path": "b.png", "species": "Flavi", "isolate_id": "", "modality": "colony", "source": "openfungi", "group_id": ""},
    ])
    with pytest.raises(ValueError, match="b.png"):
        load_manifest(path)


def test_ungrouped_openfungi_rows_become_one_group_per_image_only_when_allowed(tmp_path):
    path = _write(tmp_path, [
        {"image_path": "a.png", "species": "Flavi", "isolate_id": "", "modality": "colony", "source": "openfungi"},
        {"image_path": "b.png", "species": "Flavi", "isolate_id": "", "modality": "colony", "source": "openfungi"},
    ])
    df = load_manifest(path, allow_ungrouped=True)
    assert df["group"].nunique() == 2


def test_new_metadata_columns_default_when_absent(tmp_path):
    path = _write(tmp_path, [
        {"image_path": "a.png", "species": "S1", "isolate_id": "I1", "modality": "microscopic"},
    ])
    row = load_manifest(path).iloc[0]
    assert row["phase"] == "na"
    assert (row["genus"], row["temperature"], row["fov_id"], row["sha256"], row["split"], row["fold"]) == ("", "", "", "", "", "")
    assert pd.isna(row["z_index"])


def test_phase_must_be_mold_yeast_or_na(tmp_path):
    path = _write(tmp_path, [
        {"image_path": "a.png", "species": "S1", "isolate_id": "I1", "modality": "microscopic", "phase": "larva"},
    ])
    with pytest.raises(ValueError, match="phase"):
        load_manifest(path)
