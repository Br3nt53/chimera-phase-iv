import argparse, os, json, time, yaml, random

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def dry_run_train(experiment, model, data, seed, out_dir):
    random.seed(seed)
    rmse = round(0.1 + (seed * 0.007) + (0.01 if model.get("type") == "fractal_unet" else 0.02), 4)
    id_final = round(12.0 - (0.2 if model.get("type") == "fractal_unet" else 0.0) + (seed*0.03), 3)
    spectra_highk_err = round(0.08 + (0.01 if model.get("type") == "unet" else 0.007) + seed*0.002, 4)
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": experiment.get("name", "E1"),
        "model_name": model.get("name", "UnknownModel"),
        "model_type": model.get("type", "unknown"),
        "seed": seed,
        "rmse_placeholder": rmse,
        "id_final_placeholder": id_final,
        "spectra_highk_err_placeholder": spectra_highk_err
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_cfd2d] Dry-run wrote {out_dir}/metrics.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with open(args.experiment_config, "r") as f:
        experiment = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model = yaml.safe_load(f)
    with open(args.data_config, "r") as f:
        data = yaml.safe_load(f)
    out_root = experiment["logging"]["out_dir"]
    model_name = model.get("name", "UnknownModel")
    run_dir = os.path.join(out_root, f"{model_name}_seed{args.seed}")
    ensure_dir(run_dir)
    with open(os.path.join(run_dir, "resolved_experiment.yaml"), "w") as f:
        yaml.safe_dump(experiment, f)
    with open(os.path.join(run_dir, "resolved_model.yaml"), "w") as f:
        yaml.safe_dump(model, f)
    with open(os.path.join(run_dir, "resolved_data.yaml"), "w") as f:
        yaml.safe_dump(data, f)
    if args.dry_run:
        dry_run_train(experiment, model, data, args.seed, run_dir)
    else:
        # TODO: replace this with real training/evaluation to produce metrics.json
        dry_run_train(experiment, model, data, args.seed, run_dir)

if __name__ == "__main__":
    main()
