"""What a run was made from: code commit, data hashes, resolved config and package versions.

Every training and evaluation run writes this block into its metrics.json. With
`tracking = "mlflow"` the same params, metrics and report files also go to MLflow,
which is an optional extra: `pip install "mycoscan[mlflow]"`.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from types import ModuleType

PACKAGES = ("torch", "torchvision", "timm", "numpy", "pandas", "scipy", "pillow", "matplotlib", "mlflow")
CODE_DIR = Path(__file__).resolve().parent


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(path: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state(path: str | Path = CODE_DIR) -> dict:
    """Commit of the repository holding `path`, and whether any tracked or untracked file differs from it.
    Both are None outside a git checkout (an installed package) or without git."""
    commit = _git(Path(path), "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git(Path(path), "status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else status != ""}


def package_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def collect(manifest: str | Path, config: dict | None = None, splits: str | Path | None = None,
            checkpoint: str | Path | None = None) -> dict:
    block = {**git_state(), "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
             "splits_sha256": sha256_file(splits) if splits else None}
    if checkpoint is not None:
        block |= {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)}
    return {**block, "config": config, "packages": package_versions()}


def require_mlflow() -> ModuleType:
    try:
        import mlflow
    except ImportError as e:
        raise ImportError('tracking = "mlflow" needs the optional extra: pip install "mycoscan[mlflow]"') from e
    return mlflow


def log_to_mlflow(run_dir: Path, run_name: str, metrics: dict) -> None:
    """Params: the resolved config and the provenance hashes. Metrics: image- and isolate-level
    accuracy and macro averages. Artifacts: the report files (checkpoints stay on disk)."""
    mlflow = require_mlflow()
    prov = metrics["provenance"]
    params = {k: str(v) for k, v in (prov["config"] or {}).items()}
    params |= {k: str(prov[k]) for k in ("commit", "dirty", "manifest_sha256", "splits_sha256")}
    scores = {}
    for level in ("image_level", "isolate_level"):
        prefix = level.split("_")[0]
        scores[f"{prefix}_accuracy"] = metrics[level]["accuracy"]
        scores |= {f"{prefix}_macro_{k}": v for k, v in metrics[level]["macro"].items()}
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(scores)
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.suffix != ".pt":
                mlflow.log_artifact(str(path))
