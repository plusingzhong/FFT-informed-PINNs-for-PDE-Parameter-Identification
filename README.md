# FFT-guided Fourier Feature PINNs for Inverse Parameter Identification

Code and data (available upon request) accompanying the manuscript titled "FFT-guided Fourier feature physics-informed neural networks for inverse parameter identification of partial differential equations from scarce data"

A physics-informed neural network (PINN) framework that integrates **FFT-guided Fourier features** with **Neural Tangent Kernel (NTK) adaptive weighting** for identifying unknown parameters in partial differential equations (PDEs) from scarce data.

## Model Variants

Three architectures are provided, each controllable via two flags (`log_NTK` and `update_weights`) to enable or disable the NTK-based adaptive loss weighting:

| Variant | Architecture | NTK Adaptive Weights | `log_NTK` | `update_weights` |
|---------|-------------|----------------------|-----------|-----------------|
| PINN | Standard MLP | No | `False` | `False` |
| PINN-NTK | Standard MLP | Yes | `True` | `True` |
| RFF-PINN | Random Fourier Features | No | `False` | `False` |
| RFF-NTK | Random Fourier Features | Yes | `True` | `True` |
| FFT-PINN | FFT-guided Fourier Features | No | `False` | `False` |
| FFT-NTK | FFT-guided Fourier Features | Yes | `True` | `True` |

> **Note:** The default flag values in each script correspond to the full model (with NTK). To run ablation variants, modify `log_NTK` and `update_weights` in the `train()` call at the bottom of each script.

## Environment & Dependencies

This project is built on **TensorFlow 1.x**.

| Dependency | Version |
|------------|---------|
| Python | 3.7 |
| TensorFlow | 1.15 |
| NumPy | — |
| SciPy | — |
| Matplotlib | — |
| pyDOE | — |

### Recommended Setup

```bash
conda create -n fft-pinn python=3.7 -y
conda activate fft-pinn
pip install tensorflow==1.15 numpy scipy matplotlib pyDOE
```

## Usage Note

The data file paths in some scripts are hardcoded as absolute paths (e.g., `D:\FFT-PINN\FFT-PINN\FHN\FHN.mat`). Before running, update these paths in the `__main__` block of each script to match your local directory structure.

## Data Synthesis

The `Maxwell/MAXWELL.py` script generates Maxwell's equations simulation data via the FDTD method and performs FFT-based frequency analysis.

The distributed `.npz` files contain **noise-free** solutions only. Although `MAXWELL.py` applies Gaussian noise internally during its FFT spectral analysis stage, the saved data files store the clean, uncorrupted fields. Noise injection is instead performed dynamically at training time — each model script (e.g., `PINN.py`, `RFF_PINN.py`, `FFT_PINN.py`) adds noise to the loaded data during initialization via a configurable `noise_level` parameter. This separation allows users to freely adjust the noise intensity for different experiments without re-running the data synthesis pipeline.