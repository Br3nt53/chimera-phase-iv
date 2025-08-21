import argparse, json
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.summary, "r") as f:
        summary = json.load(f)
    models, values = [], []
    for m, agg in summary.get("by_model", {}).items():
        models.append(m); values.append(agg.get("rmse_mean", 0.0))
    plt.figure()
    plt.bar(models, values)
    plt.title("E1 placeholder: RMSE mean by model (dry-run)")
    plt.xlabel("Model"); plt.ylabel("RMSE (placeholder)"); plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"[generate_plots_stub] Wrote {args.out}")

if __name__ == "__main__":
    main()
