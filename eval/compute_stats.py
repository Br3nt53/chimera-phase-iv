import argparse, json, math, numpy as np
from scipy import stats

def cohen_d(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = ((na-1)*sa + (nb-1)*sb) / (na + nb - 2) if na+nb-2>0 else 0
    return (np.mean(a) - np.mean(b)) / math.sqrt(sp+1e-12) if sp>0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()
    with open(args.in_path, "r") as f:
        summary = json.load(f)
    # Placeholder: compute d on rmse_mean
    bm = summary.get("by_model", {}).get("UNetSmall", {}).get("rmse_mean")
    fm = summary.get("by_model", {}).get("FractalUNetSmall", {}).get("rmse_mean")
    out = {}
    if bm is not None and fm is not None:
        out["cohens_d_rmse_mean_placeholder"] = cohen_d([bm], [fm])
    rho, p = stats.spearmanr([1,2,3], [3,2,1])
    out["spearman_demo"] = {"rho": float(rho), "p": float(p)}
    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[compute_stats] Wrote {args.out_path}")

if __name__ == "__main__":
    main()
