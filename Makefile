SHELL := /bin/bash

.PHONY: help check parity dryrun train_e1 probes aggregate figures full_e1 clean verify_hashes hash_missing

help:
	@echo "Targets:"
	@echo "  check          - Verify manifests and param/FLOP parity"
	@echo "  parity         - Run Torch param/MAC parity gate (±5%)"
	@echo "  dryrun         - Run dry-run grid (3 seeds × 2 models)"
	@echo "  train_e1       - Train E1 (IND) for both models, seeds [0,1,2]"
	@echo "  probes         - (Placeholder) Run probes on saved activations"
	@echo "  aggregate      - Aggregate metrics into summary JSON + stats"
	@echo "  figures        - Render placeholder figures from summary"
	@echo "  full_e1        - check → train_e1 → aggregate → figures"
	@echo "  verify_hashes  - Verify that data manifests have correct sha256"
	@echo "  hash_missing   - Write sha256 where missing (review diffs before commit)"

check: verify_hashes parity

parity:
	python scripts/assert_param_flops_torch.py \
	  --fractal configs/models/fractal_unet_small.yaml \
	  --baseline configs/models/unet_small.yaml \
	  --tol 0.05

dryrun:
	bash scripts/run_grid.sh
	python scripts/collect_metrics.py --root artifacts/E1 --out artifacts/E1_summary.json
	python eval/generate_plots_stub.py --summary artifacts/E1_summary.json --out artifacts/fig1_ind_ood.png

train_e1:
	python run_hydra.py model=unet_small data=cfd2d_ind_v1 experiment=E1_cfd2d dry_run=false seeds_override=[0,1,2]
	python run_hydra.py model=fractal_unet_small data=cfd2d_ind_v1 experiment=E1_cfd2d dry_run=false seeds_override=[0,1,2]

probes:
	@echo "TODO: implement probe runner to load saved activations and write probes.json"

aggregate:
	python scripts/collect_metrics.py --root artifacts/E1 --out artifacts/E1_summary.json
	python eval/compute_stats.py --in artifacts/E1_summary.json --out artifacts/E1_results.json

figures:
	python eval/generate_plots_stub.py --summary artifacts/E1_summary.json --out artifacts/fig1_ind_ood.png

full_e1: check train_e1 aggregate figures

verify_hashes:
	python scripts/verify_manifests.py --verify \
	  configs/data/cfd2d_ind_v1.yaml configs/data/cfd2d_ood_v1.yaml

hash_missing:
	python scripts/verify_manifests.py --write-missing \
	  configs/data/cfd2d_ind_v1.yaml configs/data/cfd2d_ood_v1.yaml

clean:
	rm -rf artifacts/E1
