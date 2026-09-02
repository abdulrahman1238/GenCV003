# GenCV003 — VAE & DDPM from Scratch

Implementation and comparison of a **Variational Autoencoder (VAE)** and a **Denoising Diffusion Probabilistic Model (DDPM)** trained on CIFAR-10.

## Project Structure

```text
GenCV003/
├── configs/              # Model and training configurations
├── src/
│   ├── data/             # Dataset utilities
│   ├── models/           # VAE and DDPM implementations
│   ├── training/         # Training code
│   └── evaluation/       # FID, Inception Score and visualization
├── notebooks/
│   └── gencv003.ipynb    # Training, evaluation and results
    └── GenCV003_DDPM_evaluation.ipynb # contain updated evaluate result
├── pyproject.toml        # Project dependencies
├── Gencv003_report.pdf 
└── uv.lock               # Locked dependencies
```

## Reproduce the Results

### 1. Clone the repository

```bash
git clone https://github.com/abdulrahman1238/GenCV003.git
cd GenCV003
```

### 2. Install dependencies

Using `uv`:

```bash
uv sync
```

Or install the required dependencies using your preferred Python environment.

### 3. Run the notebook

Open:

```text
notebooks/gencv003.ipynb
```

Run the notebook cells in order.

The notebook contains the complete workflow for:

* Loading CIFAR-10
* Training the VAE
* Training the DDPM
* Generating samples
* Evaluating FID and Inception Score
* Visualizing generated images


## Model Weights

Trained model weights are stored in:

**[Download `weights`](https://drive.google.com/file/d/1p-VxjwIwe9lPy4GVJLkCCtaj12LNZvYS/view?usp=sharing)**

## Report

The detailed project report is available in:

**[Project Report — PDF](https://drive.google.com/file/d/1p-VxjwIwe9lPy4GVJLkCCtaj12LNZvYS/view?usp=drive_link)**


It explains the VAE and DDPM architectures, implementation, training process, evaluation, results, limitations, and the main differences between the two approaches.
