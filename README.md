<div align="center">

# ImmuVis PI-GVAE

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5?style=flat-square&logo=pytorchlightning&logoColor=white)](https://lightning.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)

**A state-of-the-art deep learning framework for immunology visualization using Graph Variational Autoencoders (PI-GVAE)**

*Built with PyTorch Lightning and Hydra configuration management for scalable immunological data analysis*

---

</div>

## 🔬 Overview

This project implements a Graph Variational Autoencoder for analyzing and visualizing Imaging Mass Cytometry (IMC) data. It provides tools for training, evaluating, and analyzing graph-based representations of immunological data.

## ✨ Features

- 🧠 **Graph VAE Architecture**: Custom implementation of Graph Variational Autoencoders with Lie group equivariance
- 📊 **IMC Data Support**: Specialized data loaders for Imaging Mass Cytometry datasets
- ⚙️ **Configurable Training**: Hydra-based configuration management for experiments
- 🚀 **Multiple Backends**: Support for CPU, GPU, and MPS training
- 📈 **Comprehensive Metrics**: Built-in reconstruction and equivariance metrics
- 📋 **Visualization Tools**: Plotting utilities for analysis and interpretation

## 🚀 Quick Start

### Prerequisites

![Python](https://img.shields.io/badge/Python-≥3.10-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-≥2.0.0-red?style=for-the-badge)
![Lightning](https://img.shields.io/badge/Lightning-≥2.0.0-purple?style=for-the-badge)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd immuvis
   ```

2. **Install dependencies** (using uv - recommended)
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install -e .
   ```

3. **Set up development tools**
   ```bash
   pre-commit install
   ```

## 💻 Usage

### Training

<details>
<summary><b>🎯 Basic Training</b></summary>

Train a model with default configuration:
```bash
uv run src/train.py
```

Alternative methods:
```bash
# Using Python directly
python src/train.py

# Using make command
make train
```

</details>

<details>
<summary><b>⚙️ Configuration</b></summary>

The project uses **Hydra** for configuration management:

| Config Type | Location | Purpose |
|-------------|----------|---------|
| `configs/train.yaml` | Main config | Training configuration |
| `configs/data/` | Data modules | Dataset configurations |
| `configs/model/` | Model arch | Architecture settings |
| `configs/trainer/` | Training setup | Hardware & optimization |

**Custom training:**
```bash
uv run src/train.py trainer=gpu logger=wandb
```

**Multi-run experiments:**
```bash
uv run src/train.py -m trainer=mps logger=wandb experiment=example hparams_search=mnist_optuna
```

</details>

<details>
<summary><b>📊 Evaluation</b></summary>

Evaluate a trained model:
```bash
uv run src/eval.py trainer=mps logger=wandb ckpt_path=path/to/checkpoint.ckpt
```

</details>

<details>
<summary><b>🧪 Testing & Quality</b></summary>

**Run tests:**
```bash
make test        # Quick tests
make test-full   # Comprehensive tests
```

**Code formatting:**
```bash
uv run ruff format
uv run ruff check --fix
```

</details>

## 📁 Project Structure

```
📦 immuvis/
├── ⚙️  configs/                 # Hydra configuration files
│   ├── 📊 data/               # Data module configs
│   ├── 🧠 model/              # Model configs
│   ├── 🏃 trainer/            # Trainer configs
│   └── 📋 train.yaml          # Main config
├── 🔬 src/
│   ├── 📊 data/              # Data modules and loaders
│   ├── 🧠 models/            # Model implementations
│   │   ├── 🔧 components/    # Model components
│   │   │   ├── 📈 metrics/   # Custom metrics
│   │   │   └── ...
│   │   └── ...
│   ├── 🛠️  utils/             # Utility functions
│   ├── 🏃 train.py          # Training script
│   └── 📊 eval.py           # Evaluation script
├── 🧪 tests/                # Test suite
└── 📝 logs/                 # Training logs and outputs
```

## 🧠 Model Architecture

| Component | Description |
|-----------|-------------|
| **Graph Transformer** | Custom implementation with Lie group equivariance |
| **Spectral Embeddings** | Graph spectral analysis components |
| **Custom Losses** | Specialized loss functions for graph VAE training |
| **Evaluation Metrics** | Reconstruction and equivariance evaluation |

## 📊 Supported Data

<table>
<tr>
<td align="center">
<img src="https://img.shields.io/badge/IMC-Supported-green?style=flat-square" alt="IMC"/>
<br/>
<b>Imaging Mass Cytometry</b>
<br/>
<i>Primary biological data</i>
</td>
<td align="center">
<img src="https://img.shields.io/badge/MNIST-Testing-blue?style=flat-square" alt="MNIST"/>
<br/>
<b>MNIST Dataset</b>
<br/>
<i>Development & testing</i>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Custom-Flexible-orange?style=flat-square" alt="Custom"/>
<br/>
<b>Custom Graphs</b>
<br/>
<i>DenseGraphBatch format</i>
</td>
</tr>
</table>

### Key Metrics Tracked

- 📈 **Reconstruction Loss**: VAE reconstruction quality
- 🔄 **KL Divergence**: Latent space regularization  
- ⚖️ **Equivariance Error**: Lie group structure preservation
- 🎯 **Graph Metrics**: Node and edge reconstruction accuracy

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{immuvis2024,
  title={ImmuVis PI-GVAE: Graph Variational Autoencoders for Immunology Visualization},
  author={[Tomasz Nocoń]},
  year={2024},
  publisher={GitHub},
  url={https://github.com/TomekNocon/IMMUVIS-PIGVAE}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

- **Author**: [Tomasz Nocoń]
- **Email**: [Tom.nocon20@gmail.com]
- **Project Link**: [https://github.com/TomekNocon/IMMUVIS-PIGVAE](https://github.com/TomekNocon/IMMUVIS-PIGVAE)

---

<div align="center">
  
**⭐ Star this repo if you found it helpful!**

Made with ❤️ for the immunology research community

</div>