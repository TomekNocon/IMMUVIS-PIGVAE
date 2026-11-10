<div align="center">

# ImmuVis PIGVAE

**Permutation-invariant graph VAE for imaging mass cytometry (IMC) crop embeddings**

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3+-ee4c2c)](https://pytorch.org)
[![Lightning](https://img.shields.io/badge/lightning-2.x-792ee5)](https://lightning.ai)
[![Hydra](https://img.shields.io/badge/config-hydra-89b8cd)](https://hydra.cc)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Code for the master thesis *[thesis title]*. Each 128 × 128 IMC crop is encoded by the
[ImmuVis](https://github.com/dav3794/multiplex-image-model) foundation model into a 16 × 16 × 768
feature map, reduced to 128 dimensions with PCA, and treated as a 16 × 16 grid graph of 256
nodes. PIGVAE, a graph VAE in the spirit of [Winter et al. (2021)](https://arxiv.org/abs/2104.09856),
learns a per-node latent and a global crop embedding (`z_global`) from this graph by
reconstruction. The embeddings are evaluated by transfer to clinical labels with gated attention
MIL (ABMIL) and explained with attention rollout and Integrated Gradients.

## Method

| Part | Final model (`experiment=pigvae`) |
|---|---|
| Input | 256 nodes × 128 PCA coefficients; 8 D4 views (rotations and flips) of every crop per batch |
| Encoder | 4-layer pre-norm transformer (width 512, 8 heads, QK-RMSNorm); each node attends to its grid neighbours (Manhattan radius 1); a one-way CLS token reads all nodes and gives `z_global` |
| Bottleneck | Per-node Gaussian latent (32-d), with the same noise across the 8 views of a crop |
| Decoder | 4-layer full-attention transformer with 2D sinusoidal positions, FiLM-conditioned on `z_global` (tanh-bounded) |
| Objective | Huber + cosine + gradient reconstruction, KL weight 0.01 annealed in from epoch 5 |
| Reference run | wandb `81zasevr`, 40 epochs |

The learned D4 permuter of the original PIGVAE is kept in the code (and on the `mnist` branch),
but the final model uses the known view order instead.

## Repository

```
configs/             Hydra configs
  experiment/          pigvae.yaml (final model), ablation/<axis>/*, baseline/dgi.yaml
  data/ model/ trainer/ callbacks/ logger/ paths/ hparams_search/
  downstream/ explain/ data_prep/ diagnostics/   configs of the pipeline scripts
src/
  data/                IMC datamodule, D4 views, PCA collation (DenseGraphBatch)
  models/              PLGraphAE (LightningModule), encoder/decoder, losses, schedulers
  downstream/          frozen-encoder feature extraction, gated ABMIL + cross-validation
  explainability/      attention rollout, Integrated Gradients, faithfulness, IMC rendering
  baselines/dgi/       Deep Graph Infomax baseline
  deploy/modal/        ONNX serving on Modal (work in progress)
  utils/               Hydra/Lightning utilities, model inspection, callbacks
  train.py eval.py     entry points
scripts/
  data/                PCA fitting, embeddings -> HDF5 conversion
  downstream/          encode -> label join -> ABMIL CV -> comparison table
  explain/             encoder / ABMIL attribution, IMC visualisation
  diagnostics/         checkpoint health report
  thesis_figures/      figure extraction (cluster) and rendering (anywhere)
  slurm/               SLURM launchers for all of the above
external/              git submodules: ImmuVis (multiplex-image-model), DGI
tests/                 unit tests + integration tests on real data
```

| Branch / tag | Contents |
|---|---|
| `main` | Final IMC model with the full pipeline, all ablation configs that need no extra code, the DGI baseline and the thesis figures |
| `mnist`, `fashion-mnist` | Toy D4 experiments on (Fashion-)MNIST, including the learned permuter |
| `ablation/*` | Ablations that need code not on `main` (CNN decoder, D4 alignment loss, no stats readout, ...) |
| `archive/*` (tags) | Earlier development snapshots |

## Setup

```bash
git clone --recurse-submodules https://github.com/TomekNocon/IMMUVIS-PIGVAE.git
cd IMMUVIS-PIGVAE
make install        # uv sync, submodules, pre-commit hooks
```

The data is not part of the repository. `configs/paths/szary.yaml` sets the root on the cluster;
the IMC datamodule expects

```
<root>/data/IMC/<dataset>/train.h5, test.h5                          # scripts/data/convert_imc_embeddings_to_h5.py
<root>/data/IMC/<dataset>/pca_model_128_center_crop_16.pkl           # scripts/data/find_pca_components.py
```

Add a file next to `szary.yaml` for another machine and select it with `paths=<name>`.

## Usage

All entry points are Hydra apps: override any config value on the command line. On the cluster,
jobs go through the launchers in `scripts/slurm/` (submit from the repository root).

**Train**

```bash
make train                                            # final model (sbatch scripts/slurm/train.sh)
make train EXPERIMENT=ablation/kl/scale_1e-3          # one ablation
bash scripts/slurm/submit_ablation.sh kl spatial      # every config of one or more ablation axes
uv run python src/train.py experiment=pigvae paths=szary trainer=gpu logger=wandb   # directly
uv run python src/train.py -m hparams_search=pigvae_optuna experiment=pigvae paths=szary trainer=gpu
```

**Evaluate**

```bash
uv run python src/eval.py experiment=pigvae paths=szary ckpt_path=<last.ckpt>
```

**Downstream transfer** (frozen encoder → per-crop features → 10-fold gated-ABMIL CV)

```bash
sbatch scripts/slurm/downstream.sh <ckpt> pigvae <run_tag>          # z_global features
sbatch scripts/slurm/downstream_sweep.sh <ckpt> pigvae <run_tag>    # raw / z_global / node features + table
```

**Explain**

```bash
sbatch --export=ALL,CKPT=<ckpt> scripts/slurm/explain.sh                  # encoder Integrated Gradients
sbatch --export=ALL,CKPT=<ckpt>,METHOD=rollout scripts/slurm/explain.sh   # attention rollout
sbatch --export=ALL,INPUT=<image.tiff> scripts/slurm/visualize_imc.sh     # marker grid + composite
```

**Diagnose a checkpoint**

```bash
uv run python scripts/diagnostics/diagnose_model.py experiment=pigvae ckpt_path=<ckpt> inspect=true
```

**Thesis figures**: `extract_*.py` reads the data and models (run on the cluster with
`scripts/slurm/run_gpu.sh` / `run_cpu.sh`); `fig_*.py` renders from the cached results into
`figures/`. Every script documents its usage in its docstring.

## Development

```bash
make lint           # ruff check + format check
make format         # all pre-commit hooks, fixing in place
make test           # fast tests
make test-full      # all tests; the integration tests need the IMC data (they skip without it)
```

## Citation

```bibtex
@mastersthesis{nocon_pigvae,
  author = {Tomasz Noco{\'n}},
  title  = {[thesis title]},
  school = {[university]},
  year   = {[year]},
  url    = {https://github.com/TomekNocon/IMMUVIS-PIGVAE}
}
```

## Acknowledgements

Built on [ImmuVis / multiplex-image-model](https://github.com/dav3794/multiplex-image-model),
[PIGVAE](https://arxiv.org/abs/2104.09856), [DGI](https://github.com/PetarV-/DGI) and the
[Lightning-Hydra template](https://github.com/ashleve/lightning-hydra-template).

## License

[MIT](LICENSE) © Tomasz Nocoń
