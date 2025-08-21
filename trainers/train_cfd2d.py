# trainers/train_cfd2d.py
import os, json, random, argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import yaml

# Import your model factory
from models.torch_models import build_from_manifest

# ---------- Data utilities ----------
def load_manifest_list(manifest_yaml_path: str):
    """Read a {files: [{path, sha256}, ...]} YAML and return the list of paths."""
    m = yaml.safe_load(Path(manifest_yaml_path).read_text())
    files = [it["path"] for it in m.get("files", [])]
    return files

class NpzPairDataset(Dataset):
    """Loads (x,y) from .npz files saved as arrays of shape [C,H,W]."""
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        d = np.load(self.paths[i])
        x = torch.from_numpy(d["x"]).float()  # [C,H,W]
        y = torch.from_numpy(d["y"]).float()  # [C,H,W]
        return x, y

def build_loaders(ind_paths, ood_paths, seed: int, batch_size: int, ind_eval_n: int = 32, ood_eval_n: int = 32):
    """Deterministic IND train/val split and small fixed eval loaders for IND/OOD."""
    g = torch.Generator().manual_seed(seed)
    n = len(ind_paths)
    idx = list(range(n))
    rng = random.Random(seed); rng.shuffle(idx)
    cut = max(1, int(0.8 * n))
    train_idx, val_idx = idx[:cut], idx[cut:]
    # Fixed eval subsets
    ind_eval_idx = val_idx[:min(ind_eval_n, len(val_idx))]
    ood_eval_idx = list(range(min(ood_eval_n, len(ood_paths))))

    train_ds = NpzPairDataset([ind_paths[i] for i in train_idx])
    ind_eval_ds = NpzPairDataset([ind_paths[i] for i in ind_eval_idx])
    ood_eval_ds = NpzPairDataset([ood_paths[i] for i in ood_eval_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    ind_eval_loader = DataLoader(ind_eval_ds, batch_size=batch_size, shuffle=False)
    ood_eval_loader = DataLoader(ood_eval_ds, batch_size=batch_size, shuffle=False)
    return train_loader, ind_eval_loader, ood_eval_loader, ind_eval_idx, ood_eval_idx

# ---------- Train / Eval ----------
def evaluate(model, loader, loss_fn):
    model.eval()
    mse_sum, mae_sum, n_obs = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            # Shapes: [B,C,H,W]
            yhat = model(x)
            mse_sum += torch.mean((yhat - y) ** 2).item() * x.size(0)
            mae_sum += torch.mean(torch.abs(yhat - y)).item() * x.size(0)
            n_obs += x.size(0)
    return {
        "mse": mse_sum / max(1, n_obs),
        "mae": mae_sum / max(1, n_obs),
    }

def train_once(
    model_cfg_path: str,
    ind_cfg_path: str,
    ood_cfg_path: str,
    out_root: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
):
    # Determinism
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load model config (YAML -> dict) and build
    model_cfg = yaml.safe_load(Path(model_cfg_path).read_text())
    model = build_from_manifest(model_cfg)
    model_name = model_cfg.get("name", Path(model_cfg_path).stem)

    # Load data manifests via data configs
    ind_cfg = yaml.safe_load(Path(ind_cfg_path).read_text())
    ood_cfg = yaml.safe_load(Path(ood_cfg_path).read_text())
    ind_manifest = ind_cfg.get("manifest_yaml", "data/manifests/IND_MANIFEST.yaml")
    ood_manifest = ood_cfg.get("manifest_yaml", "data/manifests/OOD_MANIFEST.yaml")
    ind_paths = load_manifest_list(ind_manifest)
    ood_paths = load_manifest_list(ood_manifest)
    assert len(ind_paths) > 0 and len(ood_paths) > 0, "IND/OOD manifests are empty."

    # Data
    train_loader, ind_eval_loader, ood_eval_loader, ind_eval_idx, ood_eval_idx = build_loaders(
        ind_paths, ood_paths, seed=seed, batch_size=batch_size
    )

    # Optimizer / Loss
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, n_tok = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            n_tok += x.size(0)

        ind_m = evaluate(model, ind_eval_loader, loss_fn)
        ood_m = evaluate(model, ood_eval_loader, loss_fn)
        history.append({"epoch": epoch, "loss": run_loss / max(1, n_tok), "ind_mse": ind_m["mse"], "ood_mse": ood_m["mse"]})

    # Write metrics.json
    outdir = Path(out_root) / f"{model_name}_seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "seed": seed,
        "model": model_name,
        "epochs": epochs,
        "metrics": {
            "train": history,
            "ind_eval": ind_m,
            "ood_eval": ood_m
        },
        "eval_indices": {
            "ind_val": ind_eval_idx,
            "ood_val": ood_eval_idx
        }
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train_cfd2d] Wrote {outdir/'metrics.json'}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["unet_small", "fractal_unet_small"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--ind-cfg", default="configs/data/cfd2d_ind_v1.yaml")
    ap.add_argument("--ood-cfg", default="configs/data/cfd2d_ood_v1.yaml")
    ap.add_argument("--out-root", default="artifacts/E1")
    args = ap.parse_args()

    model_cfg_path = f"configs/models/{args.model}.yaml"
    train_once(
        model_cfg_path=model_cfg_path,
        ind_cfg_path=args.ind_cfg,
        ood_cfg_path=args.ood_cfg,
        out_root=args.out_root,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

if __name__ == "__main__":
    main()
