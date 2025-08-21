import argparse, os, json, glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for path in glob.glob(os.path.join(args.root, "*", "metrics.json")):
        with open(path, "r") as f:
            rows.append(json.load(f))
    summary = {"count": len(rows), "by_model": {}}
    for r in rows:
        m = r.get("model_name", "unknown")
        d = summary["by_model"].setdefault(m, {"n": 0, "rmse_mean": 0.0})
        d["n"] += 1
        d["rmse_mean"] += r.get("rmse_placeholder", r.get("ind",{}).get("rmse_total",0.0))
    for m, d in summary["by_model"].items():
        if d["n"]:
            d["rmse_mean"] /= d["n"]
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[collect_metrics] Wrote {args.out}")

if __name__ == "__main__":
    main()
