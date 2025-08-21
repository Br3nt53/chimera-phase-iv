import subprocess, sys, yaml
import hydra

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    with open(cfg.experiment_cfg, "r") as f:
        exp = yaml.safe_load(f)
    seeds = cfg.seeds_override or exp.get("seeds", [0])
    for seed in seeds:
        cmd = [
            sys.executable, "trainers/train_cfd2d.py",
            "--experiment-config", cfg.experiment_cfg,
            "--model-config", cfg.model_cfg,
            "--data-config", cfg.data_cfg,
            "--seed", str(seed)
        ]
        if cfg.dry_run:
            cmd.append("--dry-run")
        print(f"[hydra-run] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
