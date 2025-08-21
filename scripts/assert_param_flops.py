import argparse, yaml, math, sys

def pseudo_param_count(manifest):
    ch = manifest.get("channels", [])
    ks = int(manifest.get("kernel_size", 3))
    levels = int(manifest.get("levels", len(ch)))
    total = 0
    for i in range(len(ch)-1):
        total += ch[i] * ch[i+1] * ks * ks * 2
    if manifest.get("type") == "fractal_unet":
        bf = int(manifest.get("branching_factor", 2))
        total = int(total * (1 + 0.15 * (levels-1) * (bf-1)))
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractal", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()
    with open(args.fractal, "r") as f:
        fractal = yaml.safe_load(f)
    with open(args.baseline, "r") as f:
        baseline = yaml.safe_load(f)
    pf = pseudo_param_count(fractal)
    pb = pseudo_param_count(baseline)
    rel = abs(pf - pb) / max(pb, 1)
    print(f"[assert_param_flops] pseudo_params: fractal={pf}, baseline={pb}, rel_delta={rel:.4f}")
    if rel > args.tol:
        print(f"[assert_param_flops] ERROR: Relative delta {rel:.3f} exceeds tol {args.tol:.3f}", file=sys.stderr)
        sys.exit(2)
    print("[assert_param_flops] OK within tolerance.")

if __name__ == "__main__":
    main()
