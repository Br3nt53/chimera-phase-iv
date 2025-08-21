# Chimera Phase IV — Executable Scaffold

> **Scope**: This repository scaffold operationalizes the Phase IV one-pager with concrete commands, configs, and stubs.
> It is organized so you can (1) freeze data/model manifests, (2) assert param/FLOP parity (±5%), (3) run dry-run
> grid executions to validate logging/plots, and (4) extend to real training and probes.
>
> **Neutral vs Opinion/Synthesis**: Items explicitly labeled **Neutral** mirror the Phase IV spec. Items labeled
> **Opinion/Synthesis** are pragmatic additions to make the plan executable and auditable.

---

## 0) Repo Structure (Opinion/Synthesis)

```
chimera_phase_iv_scaffold/
  README.md
  requirements.txt
  Makefile
  conf/
    config.yaml
    data/
      cfd2d_ind_v1.yaml
      cfd2d_ood_v1.yaml
    experiment/
      E1_cfd2d.yaml
    model/
      fractal_unet_small.yaml
      unet_small.yaml
  configs/
    data/
      cfd2d_ind_v1.yaml
      cfd2d_ood_v1.yaml
    experiments/
      E1_cfd2d.yaml
    models/
      fractal_unet_small.yaml
      unet_small.yaml
  data/
    manifests/
      IND_MANIFEST.yaml
      OOD_MANIFEST.yaml
  eval/
    compute_stats.py
    generate_plots_stub.py
  models/
    torch_models.py
  probes/
    id_twonn.py
    ib_knn_stub.py
    spectra.py
    patching_stub.py
    edit_rome_stub.py
  scripts/
    run_grid.sh
    collect_metrics.py
    assert_param_flops.py
    assert_param_flops_torch.py
    verify_manifests.py
  trainers/
    train_cfd2d.py
  .github/workflows/
    chimera-ci.yml
    chimera-gpu-e1.yml
  Deviation_Log.md
  Decision_Log.md
  artifacts/  # created at runtime
```

---

## 1) Quickstart (Opinion/Synthesis)

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Freeze data splits** (edit `configs/data/*.yaml` and `data/manifests/*.yaml` to point at your files).

**Param/FLOP parity gate**:

```bash
python scripts/assert_param_flops_torch.py   --fractal configs/models/fractal_unet_small.yaml   --baseline configs/models/unet_small.yaml   --tol 0.05
```

**Dry-run E1 grid**:

```bash
bash scripts/run_grid.sh
python scripts/collect_metrics.py --root artifacts/E1 --out artifacts/E1_summary.json
python eval/generate_plots_stub.py --summary artifacts/E1_summary.json --out artifacts/fig1_ind_ood.png
```

---

## 7) Continuous Integration (Opinion/Synthesis)

A minimal GitHub Actions workflow is included at `.github/workflows/chimera-ci.yml` (CPU) and a GPU matrix runner at `.github/workflows/chimera-gpu-e1.yml` (requires self-hosted GPU).

---

## 8) Logs for Governance (Opinion/Synthesis)

- **Deviation_Log.md** — append-only record of any departures from prereg.
- **Decision_Log.md** — major decisions with context and alternatives.

---

## 9) What the Torch + Hydra pieces do (Summary; Neutral vs Opinion/Synthesis)

- **Torch parity checker (Opinion/Synthesis):** exact params & MACs parity using `torchinfo`/`thop`; fails if >±5%.
- **Minimal Torch models (Opinion/Synthesis):** UNet & FractalUNet respond to manifest fields; used for parity and smoke tests.
- **Hydra entrypoint (Opinion/Synthesis):** `run_hydra.py` centralizes runs; `model=..., data=..., experiment=...` overrides.

---

## 10) One-command automation (Makefile) — Opinion/Synthesis

- `make check` — Verify data hashes and run the parity gate (±5%).
- `make dryrun` — Exercise the whole pipeline without training.
- `make train_e1` — Train baseline & fractal on IND for seeds [0,1,2].
- `make aggregate` — Collate per-seed metrics and compute basic stats.
- `make figures` — Render a placeholder figure from the summary.
- `make full_e1` — `check` → `train_e1` → `aggregate` → `figures`.

---

## 11) Automated GPU runs in CI (self-hosted) — Opinion/Synthesis

A GPU workflow at `.github/workflows/chimera-gpu-e1.yml` lets you:
- Trigger manual runs (`workflow_dispatch`) or a weekly run (cron).
- Run a **matrix** over models × seeds.
- Aggregate results and publish summary + figures as CI artifacts.

> **Note:** Register a **self-hosted GPU runner** with labels: `self-hosted, linux, x64, gpu`.

---

## 12) Data governance automation — Opinion/Synthesis

- `make verify_hashes` — fail if any manifest hash deviates.
- `make hash_missing` — compute and write sha256 for entries with `sha256: null`.

---

## 13) Stats automation — Opinion/Synthesis

`eval/compute_stats.py` wires Cohen’s d, Spearman ρ, BCa CIs, and BH correction. Replace placeholders with your per-seed arrays to produce preregistered stats in one step.
