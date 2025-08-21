import json
import subprocess
import sys
from pathlib import Path

def test_research_cli_train(tmp_path):
    out_root = tmp_path / "run"
    cmd = [
        sys.executable,
        "scripts/research_cli.py",
        "train",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--out-root",
        str(out_root),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    metrics_files = list(out_root.glob("*/metrics.json"))
    assert metrics_files, "metrics.json not found"
    data = json.loads(metrics_files[0].read_text())
    assert "metrics" in data and "train" in data["metrics"], "metrics content missing"
