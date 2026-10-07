import pytest

from mycoscan.synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_manifest(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("syn"), size=64, fovs_per_device=1, openfungi_per_genus=4)
