"""Saving and loading trained models.

SECURITY: joblib/pickle files can execute arbitrary code when loaded. Only load artifacts
produced by this project. `load_model` refuses files whose SHA-256 does not match the
manifest written at training time, which catches accidental or malicious modification of
committed artifacts (it does not make loading an untrusted manifest safe).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from .config import ARTIFACTS_DIR

MANIFEST_NAME = "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(bundle: dict[str, Any], name: str, models_dir: Path | None = None) -> dict[str, Any]:
    """Persist a model bundle and return its manifest entry."""
    models_dir = Path(models_dir or ARTIFACTS_DIR / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(bundle, path, compress=3)
    return {"file": f"models/{path.name}", "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def write_manifest(entries: dict[str, Any], extra: dict[str, Any], artifacts_dir: Path | None = None) -> Path:
    artifacts_dir = Path(artifacts_dir or ARTIFACTS_DIR)
    path = artifacts_dir / MANIFEST_NAME
    path.write_text(json.dumps({"models": entries, **extra}, indent=2, sort_keys=True))
    return path


def load_manifest(artifacts_dir: Path | None = None) -> dict[str, Any]:
    path = Path(artifacts_dir or ARTIFACTS_DIR) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m ml.src.train` first.")
    return json.loads(path.read_text())


def load_model(name: str, artifacts_dir: Path | None = None) -> dict[str, Any]:
    """Load a trusted model bundle after verifying its checksum against the manifest."""
    artifacts_dir = Path(artifacts_dir or ARTIFACTS_DIR)
    entry = load_manifest(artifacts_dir)["models"].get(name)
    if entry is None:
        raise KeyError(f"model {name!r} is not in the manifest")
    path = artifacts_dir / entry["file"]
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"checksum mismatch for {path.name}; refusing to load")
    return joblib.load(path)
