import hashlib

import numpy as np
from PIL import Image

from mycoscan.manifest import load_manifest
from mycoscan.splits import make_folds


def test_synthetic_manifest_loads_with_new_metadata(synthetic_manifest):
    df = load_manifest(synthetic_manifest)
    cmu = df[df["source"] == "cmu"]
    assert (cmu["genus"] != "").all()
    micro = cmu[cmu["modality"] == "microscopic"]
    assert (micro["fov_id"] != "").all()
    assert micro["z_index"].notna().all()
    first = df.iloc[0]
    assert first["sha256"] == hashlib.sha256(open(first["image_path"], "rb").read()).hexdigest()


def test_openfungi_images_come_in_near_duplicate_groups(synthetic_manifest):
    of = load_manifest(synthetic_manifest)
    of = of[of["source"] == "openfungi"]
    sizes = of.groupby("group_id").size()
    assert (sizes >= 2).all()
    a, b = of[of["group_id"] == sizes.index[0]]["image_path"].iloc[:2]
    pa, pb = (np.asarray(Image.open(p).convert("RGB"), dtype=float) for p in (a, b))
    assert pa.shape == pb.shape
    assert not np.array_equal(pa, pb)
    assert np.abs(pa - pb).mean() < 40


def test_dimorphic_classes_have_both_phases_and_others_none(synthetic_manifest):
    df = load_manifest(synthetic_manifest)
    micro = df[(df["source"] == "cmu") & (df["modality"] == "microscopic")]
    phases = micro.groupby("species")["phase"].agg(lambda s: tuple(sorted(set(s))))
    assert phases["Talaromyces_marneffei"] == ("mold", "yeast")
    assert phases["Sporothrix_schenckii_complex"] == ("mold", "yeast")
    assert phases["Aspergillus_fumigatus"] == ("mold",)


def test_planted_duplicates_never_straddle_a_grouped_fold(synthetic_manifest):
    of = load_manifest(synthetic_manifest)
    of = of[of["source"] == "openfungi"].reset_index(drop=True)
    for fold in make_folds(of, "kfold", n_folds=3, seed=0):
        assert set(of["group"].iloc[fold.train_idx]) & set(of["group"].iloc[fold.val_idx]) == set()
