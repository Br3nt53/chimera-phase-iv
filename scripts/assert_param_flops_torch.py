import argparse, yaml, torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from torchinfo import summary
from thop import profile
from models.torch_models import build_from_manifest

def load_manifest(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractal", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()
    mf = load_manifest(args.fractal)
    mb = load_manifest(args.baseline)
    mod_f = build_from_manifest(mf)
    mod_b = build_from_manifest(mb)
    input_shape_f = tuple(mf.get("input_shape", [1, 128, 128]))
    input_shape_b = tuple(mb.get("input_shape", [1, 128, 128]))
    assert input_shape_f == input_shape_b, "Input shapes must match."
    bs = 1
    dummy = torch.randn(bs, *input_shape_f)
    pf = sum(p.numel() for p in mod_f.parameters())
    pb = sum(p.numel() for p in mod_b.parameters())
    macs_f, _ = profile(mod_f, inputs=(dummy,), verbose=False)
    macs_b, _ = profile(mod_b, inputs=(dummy,), verbose=False)
    rel_params = abs(pf - pb) / max(pb, 1)
    rel_macs = abs(macs_f - macs_b) / max(macs_b, 1)
    print(f"[assert_param_flops_torch] params: fractal={pf:,} baseline={pb:,} rel_delta={rel_params:.4f}")
    print(f"[assert_param_flops_torch] macs:   fractal={macs_f:,} baseline={macs_b:,} rel_delta={rel_macs:.4f}")
    ok = (rel_params <= args.tol) and (rel_macs <= args.tol)
    if not ok:
        raise SystemExit(f"Parity failed: params Δ={rel_params:.3f}, macs Δ={rel_macs:.3f} > tol {args.tol:.3f}")
    print("[assert_param_flops_torch] OK within tolerance.")

if __name__ == "__main__":
    main()
