# ImmuVis PIGVAE: Fashion-MNIST toy experiment

This branch holds the Fashion-MNIST experiment of the thesis: a controlled setting in which the D4
symmetry handled by PIGVAE can be checked by eye. The main IMC model, the full pipeline and the
documentation are on [`main`](https://github.com/TomekNocon/IMMUVIS-PIGVAE/tree/main); the
other toy dataset is on [`mnist`](https://github.com/TomekNocon/IMMUVIS-PIGVAE/tree/mnist).

Each 24 x 24 image is cut into a 6 x 6 grid of 4 x 4 patches (36 nodes, 16 pixel features
each) and shown in all 8 D4 views (rotations and flips). Unlike the final IMC model, the learned
permuter is active: it predicts which D4 transform produced each view.

## Setup

```bash
git clone -b fashion-mnist https://github.com/TomekNocon/IMMUVIS-PIGVAE.git
cd IMMUVIS-PIGVAE
make install        # uv sync + pre-commit hooks
```

Fashion-MNIST is downloaded by torchvision into `data/` on first use.

## Usage

```bash
make train ARGS="experiment=example"                     # train locally (data=fashion_mnist)
make train ARGS="experiment=example trainer=gpu logger=wandb"
uv run python src/eval.py experiment=example ckpt_path=<last.ckpt>
make test
```

Thesis reruns on the cluster use `scripts/rerun.sbatch` (see its header).

## License

[MIT](https://github.com/TomekNocon/IMMUVIS-PIGVAE/blob/main/LICENSE) © Tomasz Nocoń
