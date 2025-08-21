#!/usr/bin/env python3
"""Generate tiny CFD2D sample data and manifests.

The original repository shipped small ``.npz`` files for quick
experimentation.  Binary files are avoided in this repo; this script
recreates them on demand and writes manifests with their sha256 hashes.
"""
from pathlib import Path
import hashlib
import numpy as np
import yaml

# Deterministic toy arrays for four samples
DATA = {
    "ind_sample_00.npz": {
        "x": np.zeros((1, 8, 8), dtype=np.float32),
        "y": np.ones((1, 8, 8), dtype=np.float32),
    },
    "ind_sample_01.npz": {
        "x": np.full((1, 8, 8), 0.5, dtype=np.float32),
        "y": np.full((1, 8, 8), -0.5, dtype=np.float32),
    },
    "ood_sample_00.npz": {
        "x": np.full((1, 8, 8), -1.0, dtype=np.float32),
        "y": np.full((1, 8, 8), 1.0, dtype=np.float32),
    },
    "ood_sample_01.npz": {
        "x": np.linspace(0, 1, 64, dtype=np.float32).reshape(1, 8, 8),
        "y": np.linspace(1, 0, 64, dtype=np.float32).reshape(1, 8, 8),
    },
}

IND_FILES = ["ind_sample_00.npz", "ind_sample_01.npz"]
OOD_FILES = ["ood_sample_00.npz", "ood_sample_01.npz"]

def ensure_sample_data(root: Path = Path("data/samples")) -> None:
    """Create sample npz files and their manifests if missing."""
    root.mkdir(parents=True, exist_ok=True)
    manifest_root = Path("data/manifests")
    manifest_root.mkdir(parents=True, exist_ok=True)

    # Write sample npz files
    for fname, arrays in DATA.items():
        fpath = root / fname
        if not fpath.exists():
            np.savez(fpath, **arrays)

    # Helper to emit manifest YAML
    def write_manifest(names, manifest_path):
        entries = []
        for fname in names:
            fpath = root / fname
            h = hashlib.sha256(fpath.read_bytes()).hexdigest()
            entries.append({"path": str(fpath), "sha256": h})
        manifest_path.write_text(yaml.safe_dump({"files": entries}, sort_keys=False))

    write_manifest(IND_FILES, manifest_root / "IND_MANIFEST.yaml")
    write_manifest(OOD_FILES, manifest_root / "OOD_MANIFEST.yaml")

if __name__ == "__main__":
    ensure_sample_data()
    print("Generated sample CFD2D data and manifests.")
