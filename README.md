# ImmuVis PI-GVAE

A deep learning framework for immunology visualization using Graph Variational Autoencoders (PI-GVAE) built with PyTorch Lightning and Hydra configuration management.

## Overview

This project implements a Graph Variational Autoencoder for analyzing and visualizing Imaging Mass Cytometry (IMC) data. It provides tools for training, evaluating, and analyzing graph-based representations of immunological data.

## Features

- **Graph VAE Architecture**: Custom implementation of Graph Variational Autoencoders with Lie group equivariance
- **IMC Data Support**: Specialized data loaders for Imaging Mass Cytometry datasets
- **Configurable Training**: Hydra-based configuration management for experiments
- **Multiple Backends**: Support for CPU, GPU, and MPS training
- **Comprehensive Metrics**: Built-in reconstruction and equivariance metrics
- **Visualization Tools**: Plotting utilities for analysis and interpretation

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- Lightning >= 2.0.0

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd immuvis
```

2. Install dependencies using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Usage

### Training

Train a model with default configuration:
```bash
uv run src/train.py
```

Or use:
```bash
python src/train.py
```

Or use the make command:
```bash
make train
```

### Configuration

The project uses Hydra for configuration management. Key configuration files:
- `configs/train.yaml`: Main training configuration
- `configs/data/`: Data module configurations
- `configs/model/`: Model architecture configurations
- `configs/trainer/`: Training setup configurations

Example training with custom configuration:
```bash
uv run src/train.py trainer=gpu logger=wandb
```

And for multirun:
```bash
uv run src/train.py -m trainer=mps logger=wandb experiment=example hparams_search=mnist_optuna
```

### Evaluation

Evaluate a trained model:
```bash
uv run src/eval.py trainer=mps logger=wandb ckpt_path=path/to/checkpoint.ckpt
```

### Testing

Run tests:
```bash
make test        # Run quick tests
make test-full   # Run all tests
```

### Formatting

```bash
uv run ruff format
uv run ruff check --fix
```

## Project Structure

```
immuvis/
├── configs/                 # Hydra configuration files
│   ├── data/               # Data module configs
│   ├── model/              # Model configs
│   ├── trainer/            # Trainer configs
│   └── train.yaml          # Main config
├── src/
│   ├── data/              # Data modules and loaders
│   ├── models/            # Model implementations
│   │   ├── components/    # Model components
│   │   │   ├── metrics/   # Custom metrics
│   │   │   └── ...
│   │   └── ...
│   ├── utils/             # Utility functions
│   ├── train.py          # Training script
│   └── eval.py           # Evaluation script
├── tests/                # Test suite
└── logs/                 # Training logs and outputs
```

## Model Components

- **Custom Graph Transformer**: Implementation with Lie group equivariance
- **Spectral Embeddings**: Graph spectral analysis components
- **Custom Losses**: Specialized loss functions for graph VAE training
- **Metrics**: Reconstruction and equivariance evaluation metrics

## Data

The framework supports:
- IMC (Imaging Mass Cytometry) data
- MNIST data (for testing)
- Custom graph datasets through the DenseGraphBatch format

## Development

### Code Quality

Format code:
```bash
make format
```

Clean generated files:
```bash
make clean
```

### Contributing

1. Create a feature branch
2. Make changes
3. Run tests and formatting
4. Submit a pull request

## License

[Add your license information here]

## Citation

[Add citation information if applicable]

## Contact

[Add contact information]