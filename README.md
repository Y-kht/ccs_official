# Contrastive Conformal Sets (CCS)

Official code for **"Contrastive Conformal Sets"**.

CCS constructs conformal covering sets directly in the semantic feature space of a
(frozen) pretrained encoder. The sets include positive samples of an anchor at any
user-specified coverage level (a distribution-free conformal guarantee) while their
learnable geometry — a single-norm metric or a generalized hyper-ball — is optimized
to maximize the exclusion of negative samples. The learned sets can also be used
off-the-shelf for OOD detection.

## File Structure

```
ccs_official/
├── configs/                        # JSON configs reproducing the paper experiments
│   ├── config_simulations.json         # 3D simulation (Table 1)
│   ├── config_c_rep_a005.json          # CIFAR100 coverage (Table 2)
│   ├── config_stl10_res_a005.json      # STL10 coverage (appendix)
│   ├── config_im_res_a005.json         # ImageNet100 coverage (appendix)
│   ├── config_c_rep_a005_HIB.json      # Convex-hull / HIB / HIB-MoG comparison
│   ├── config_c_rep_a005_sensitivity.json  # Sensitivity analysis (appendix)
│   ├── config_ood_c_rep.json           # OOD: CIFAR100 (ID) vs SVHN, CIFAR10 (Table 3)
│   ├── config_ood_stl10_resnet18.json  # OOD: STL10 (ID) vs SVHN, CIFAR10/100 (appendix)
├── utils.py                        # Core library: data, CCS model, losses, training, baselines
├── simulation_utils.py             # Simulation counterpart (no encoder) + covering-set plots
├── tables_coverage.py              # Coverage/exclusion/volume experiments on image datasets
├── tables_simulation.py            # Coverage/exclusion/volume experiments on simulated clusters
├── tables_ood.py                   # OOD detection experiments (CCS + baselines)
├── sensitivity_analysis.py         # Hyperparameter sensitivity sweeps
├── ood_baselines_utils.py          # SOTA OOD baselines (COMBOOD, D-KNN, CIDER, ViM, NAC)
├── results/                        # Pre-computed coverage tables + checkpoints directory
├── results_ood/                    # Pre-computed OOD tables
├── results_sim/                    # Pre-computed simulation table + figures
├── environment.yml                 # Conda environment
└── LICENSE
```

## Setup

```bash
conda env create -f environment.yml
conda activate cu121
```

**Data.** CIFAR100, CIFAR10, STL10, SVHN, and DTD download automatically into
`./data`. NINCO is fetched through `pytorch-ood`. Two assets must be provided
manually:

- **ImageNet100** (100-class subset of ImageNet-1K, e.g. from Kaggle): pass its
  train/val class folders via `--train_dir`/`--val_dir` (coverage) or set
  `"imagenet100_root"` in the OOD config.
- **SimCLR STL10 checkpoint** (`resnet18_simclr` backbone): set the environment
  variable `SIMCLR_CKPT=/path/to/ResNet_simclr_STL10.ckpt`.

The RepVGG-A2 (CIFAR100) and ResNet-18 (ImageNet-1K) backbones download
automatically.

## Running the Experiments

### 1. Coverage / exclusion / volume (image datasets)

```bash
# CIFAR100 (Table 2)
python tables_coverage.py --config configs/config_c_rep_a005.json \
    --seeds 42 43 44 45 46 47 --note RUN1

# STL10 / ImageNet100 (appendix tables)
python tables_coverage.py --config configs/config_stl10_res_a005.json --seeds 42 43 44 45 46 47 --note RUN1
python tables_coverage.py --config configs/config_im_res_a005.json  --seeds 42 43 44 45 46 47 --note RUN1 \
    --train_dir path/to/imagenet-100/train --val_dir path/to/imagenet-100/val

# Convex hull / HIB / HIB-MoG comparison
python tables_coverage.py --config configs/config_c_rep_a005_HIB.json --seeds 42 43 44 45 46 47 --note RUN1
```

Outputs: a LaTeX table and a JSON summary in `results/`, and one checkpoint per
(method, seed) under `results/checkpoints/<method-tag>_<note>/`. Checkpoint file
names encode the hyperparameters; rerunning with the same config and `--note`
reuses existing checkpoints instead of retraining.

### 2. OOD detection

OOD evaluation **loads the checkpoints trained by `tables_coverage.py`**, so run
the matching coverage config first with the same `--note`:

```bash
python tables_ood.py --config configs/config_ood_c_rep.json --note RUN1          # Table 3
python tables_ood.py --config configs/config_ood_stl10_resnet18.json --note RUN1 # appendix
```

Outputs: a LaTeX table and JSON in the config's `save_dir` (e.g. `results_ood/cifar100/`).

### 3. Simulation

```bash
python tables_simulation.py --config configs/config_simulations.json \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --note RUN1
```

The cluster data is generated on the first seed and held fixed across seeds.
Setting `"plot": true` in the config additionally renders an exact covering-set
figure per CCS variant (for one random test anchor) into `results_sim/figures/`.

### 4. Sensitivity analysis

```bash
python sensitivity_analysis.py --config configs/config_c_rep_a005_sensitivity.json --seeds 42
```

Sweeps α, sigmoid steepness T, InfoNCE weight and temperature, volume weight λ,
and the per-anchor sample count k; prints LaTeX-friendly (value, metric) columns.

## Command-Line Arguments

| Argument | Scripts | Meaning |
|---|---|---|
| `--config` | all | Path to the experiment JSON config |
| `--seeds` | coverage, simulation, sensitivity, ood | Random seeds (space-separated). For `tables_ood.py` this *overrides* the seeds listed in the config |
| `--note` | coverage, simulation, ood | Free-form tag appended to checkpoint/table names; use the **same note** for a coverage run and its OOD run |
| `--save_dir` | coverage, simulation, sensitivity, ood | Output directory (OOD default comes from the config) |
| `--num_gpus` | coverage, sensitivity | Number of GPUs for `DataParallel` (default: all visible) |
| `--train_dir`, `--val_dir` | coverage, sensitivity | ImageNet100 train/val folders (only needed for `imagenet100`) |

## Config Reference

### Coverage / simulation configs (`shared` block)

| Key | Meaning |
|---|---|
| `dataset`, `model` | `cifar100`/`stl10`/`imagenet100` with `repvgg_a2`/`resnet18_simclr`/`resnet18` |
| `alpha` | Miscoverage tolerance (target coverage = 1 − α) |
| `k` | Number of positive and negative samples per anchor |
| `r` | Fraction of the training split used to learn the set geometry |
| `sampling` | `augmentation` (unsupervised) or `label` (supervised) positive/negative sampling |
| `batch_size`, `num_workers`, `optimizer`, `grad_clip` | Standard training knobs (SGD uses momentum 0.9, weight decay 5e-4, cosine annealing) |
| `scale` | Temperature T of the sigmoid surrogate in the negative-exclusion loss |
| `reg_w` | Weight of the optional (m, p) stability regularizer (0 in the paper) |
| `n_norms` | Projector output dimension flag for InfoNCE variants (1 = keep feature dimension) |
| `use_amp` | Mixed-precision training (kept `false` in the paper: fp16 overflows the single-norm metric) |
| `with_replacement` | Sample positives/negatives with replacement (label sampling) |
| `contrastive_weight`, `nce_temperature`, `delta` | InfoNCE loss weight λ_InfoNCE, temperature τ, and margin buffer |
| `volume_weight` | Weight of the volume term in the alternating InfoNCE phase |
| **simulation only:** `d`, `n_clusters`, `n_per_cluster`, `separation`, `std`, `distribution` | Cluster geometry (`mixed` = anisotropic Gaussians + 3D crescents) |
| `train_ratio` | Train fraction of the simulated data (rest is calibration + test) |
| `p_train`, `p_test` | Probability that a "positive" is drawn from the anchor's own cluster (train/calibration vs. test); enables controlled contamination experiments |
| `shift` | If `true`, additionally report class-consistent coverage (Cov_class) |
| `plot` | If `true`, render exact covering-set figures (simulation only) |

### Method entries (`methods` list)

| Key | Meaning |
|---|---|
| `name` | Row name in the output table |
| `type` | `baseline` (closed-form) or `metric` (learned CCS); `ccs` in OOD configs |
| `case` | Baseline type: `0a` (ℓ2 ball), `0b` (Mahalanobis), `hull`, `hib`, `mog` |
| `metric` | `single` (matrix M + scalar p) or `generalized` (per-dimension m_j, p_j) |
| `problem` | `neg` (maximize negative exclusion) or `vol` (minimize volume) |
| `lam` | If set, uses the combined objective (1−λ)·neg + λ·volume |
| `lr`, `epochs` | Learning rate and epochs for the metric parameters |
| `add_infonce`, `model_lr`, `alternating_training` | Enable the InfoNCE projector, its learning rate, and epoch-wise alternation between projector and metric updates |

### OOD configs (top level)

| Key | Meaning |
|---|---|
| `id_dataset`, `model` | In-distribution dataset and backbone |
| `ood_datasets` | Any of `svhn`, `cifar10`, `cifar100`, `dtd`/`textures`, `ninco`, `imagenet100` |
| `sampling` | Sampling used for calibration/eval features (must match the trained checkpoints) |
| `knn_k`, `n_ref` | KNN baseline neighbors and reference-set size |
| `imagenet100_root` | ImageNet100 root folder (when used as ID or OOD) |
| `seeds`, `save_dir` | Evaluation seeds and output directory |
| `methods` | CCS entries (`type: "ccs"`) must repeat the training hyperparameters (`k`, `r`, `lr`, `epochs`, `alpha`, `lam`, …) so the checkpoint file names resolve; baseline entries use `type: "msp"`, `"energy"`, `"mahalanobis"`, `"knn"`, `"combood"`, `"d_knn"`, `"cider"`, `"vim"`, `"nac"` |

## Reproducing the Paper Tables

| Paper table | Command |
|---|---|
| Simulation (Table 1) | `tables_simulation.py --config configs/config_simulations.json --seeds 0 … 19` |
| CIFAR100 coverage (Table 2) | `tables_coverage.py --config configs/config_c_rep_a005.json --seeds 42 … 47` |
| CIFAR100 OOD (Table 3) | `tables_ood.py --config configs/config_ood_c_rep.json` (after Table 2 run) |
| Hull/HIB coverage (Table 4) | `tables_coverage.py --config configs/config_c_rep_a005_HIB.json --seeds 42 … 47` |
| STL10 / ImageNet100 coverage (appendix) | `tables_coverage.py` with the corresponding config |
| STL10 OOD (appendix) | `tables_ood.py --config configs/config_ood_stl10_resnet18.json` |
| Sensitivity (appendix) | `sensitivity_analysis.py --config configs/config_c_rep_a005_sensitivity.json` |

Pre-computed outputs for all tables are included under `results/`, `results_ood/`,
and `results_sim/`.

## License

Released under the MIT License (see `LICENSE`).