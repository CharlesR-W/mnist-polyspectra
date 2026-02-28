# MNIST N-Point Correlators & Polyspectra

Compute N-point connected correlators (cumulants) on MNIST images and transform them to Fourier/wavelet polyspectra for exploratory analysis of higher-order statistical structure.

## Overview

This project analyzes the higher-order, non-Gaussian statistical structure of MNIST digit images using:
- **2-point correlators** → Power spectrum P(k)
- **3-point correlators** → Bispectrum B(k₁, k₂, k₃)
- Both **Fourier** and **wavelet** transforms
- **Per-digit class** comparisons

### What are N-point correlators?

N-point correlators measure joint statistics of pixel intensities at N spatial locations. The **connected correlators** (cumulants) isolate genuine N-point interactions by subtracting lower-order contributions:

**2-point (connected):**
```
C₂(r) = ⟨I(x) I(x+r)⟩ - ⟨I⟩²
```

**3-point (connected):**
```
C₃(r₁, r₂) = ⟨I(x) I(x+r₁) I(x+r₂)⟩ - [disconnected terms]
```

For centered images (⟨I⟩ = 0), the 3-point cumulant simplifies to the raw triple product.

### Polyspectra

Polyspectra are Fourier transforms of correlators:
- **Power spectrum**: P(k) = FFT[C₂(r)] - captures Gaussian structure
- **Bispectrum**: B(k₁,k₂,k₃) = FFT[C₃(r₁,r₂)] - captures phase coupling and non-Gaussianity

Natural images typically show:
- Power-law decay P(k) ∼ k^α with α ≈ -2 (scale invariance)
- Non-zero bispectrum (non-Gaussian, nonlinear structure)

## Installation

### Requirements
- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone/navigate to project
cd mnist-polyspectra

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Optional Development Tools

```bash
# Install dev dependencies
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

## Quick Start

### Python Script

```python
from src.config import CorrelatorConfig, SamplingConfig
from src.correlators import compute_connected_2point, compute_connected_3point
from src.polyspectra import compute_power_spectrum, fit_power_law
from src.visualization import plot_power_spectrum
from sklearn.datasets import fetch_openml
import numpy as np

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, parser='auto')
images = mnist.data.values.reshape(-1, 28, 28)[:1000]  # Subset for speed

# Configure
config = CorrelatorConfig(n_samples_per_config=500, n_configurations=200)
samp_config = SamplingConfig()

# Compute correlators
C2_result = compute_connected_2point(images, config, samp_config)
C3_result = compute_connected_3point(images, config, samp_config)

# Compute power spectrum
P_data = compute_power_spectrum(C2_result)
fit_data = fit_power_law(P_data['k_values'], P_data['P_k_radial'])

print(f"Power-law exponent: α = {fit_data['exponent']:.3f}")

# Visualize
fig = plot_power_spectrum(P_data, fit_data)
fig.savefig('power_spectrum.png')
```

### Jupyter Notebook

For full interactive analysis:

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook includes:
1. Data loading and visualization
2. Overall correlator computation
3. Fourier and wavelet polyspectra
4. Per-digit class analysis
5. Summary figures and statistics

## Mathematical Background

### Connected Correlators (Cumulants)

For N=2:
```
C₂(r) = ⟨I(x) I(x+r)⟩ - ⟨I(x)⟩²
```

For N=3:
```
C₃(r₁, r₂) = ⟨I(x) I(x+r₁) I(x+r₂)⟩
             - ⟨I(x)⟩ ⟨I(x+r₁) I(x+r₂)⟩
             - ⟨I(x+r₁)⟩ ⟨I(x) I(x+r₂)⟩
             - ⟨I(x+r₂)⟩ ⟨I(x) I(x+r₁)⟩
             + 2⟨I(x)⟩ ⟨I(x+r₁)⟩ ⟨I(x+r₂)⟩
```

### Translation Invariance

We assume correlators depend only on separation vectors, not absolute positions. This allows efficient computation via sampling:

1. Sample separation vectors (r₁, r₂)
2. For each configuration, sample anchor points x
3. Average I(x)I(x+r₁)I(x+r₂) over images and anchor points

### Bispectrum Configurations

The bispectrum satisfies k₁ + k₂ + k₃ = 0 (triangle closure). We extract canonical slices:

- **Equilateral**: |k₁| = |k₂| = |k₃| - self-similar interactions
- **Squeezed**: k₃ ≪ k₁ ≈ k₂ - coupling to long wavelengths
- **Folded**: k₁ ≈ 2k₂ ≈ 2k₃ - harmonic structure

## API Reference

### Configuration

**`CorrelatorConfig`**
- `n_points`: N for N-point correlator (2 or 3)
- `max_separation`: Maximum separation in pixels
- `n_samples_per_config`: Anchor points per configuration
- `n_configurations`: Number of (r₁, r₂) to sample
- `translation_invariant`: Assume translation invariance

**`SamplingConfig`**
- `separation_grid`: Grid of separation magnitudes
- `n_angles`: Angular samples per magnitude
- `emphasize_geometries`: ['equilateral', 'squeezed', 'collinear']
- `geometry_oversample`: Oversampling factor

**`PolyspectraConfig`**
- `use_fourier`, `use_wavelet`: Enable transforms
- `wavelet_family`: Wavelet type (e.g., 'db4')
- `n_wavelet_scales`: Decomposition levels

### Core Functions

**`correlators.compute_connected_2point(images, config, sampling_config)`**
- Returns: `{'C2': array, 'separations': array, 'errors': array}`

**`correlators.compute_connected_3point(images, config, sampling_config)`**
- Returns: `{'C3': array, 'r1_vectors': array, 'r2_vectors': array, 'errors': array}`

**`polyspectra.compute_power_spectrum(C2_data, config)`**
- Returns: `{'P_k_radial': array, 'k_values': array, 'P_k_2d': array}`

**`polyspectra.compute_bispectrum(C3_data, config)`**
- Returns: `{'B_equilateral': dict, 'B_squeezed': dict, 'B_2d': dict}`

**`polyspectra.fit_power_law(k_values, P_k)`**
- Fit P(k) ∼ k^α
- Returns: `{'exponent': α, 'amplitude': A, 'r_squared': R²}`

### Visualization

All plotting functions in `visualization.py` accept optional `ax` argument for subplot integration:

- `plot_power_spectrum(P_data, fit_data, digit, ax, config)`
- `plot_bispectrum_equilateral(B_data, digit, ax, config)`
- `plot_bispectrum_2d(B_data, digit, ax, config)`
- `plot_wavelet_spectra(wavelet_data, digit, config)`
- `plot_digit_comparison_heatmap(all_digits_data, config)`
- `create_summary_figure(P_data, B_data, wavelet_data, fit_data, digit, config)`

## Project Structure

```
mnist-polyspectra/
├── src/
│   ├── config.py           # Configuration dataclasses
│   ├── correlators.py      # N-point correlator computation
│   ├── polyspectra.py      # Fourier/wavelet transforms
│   ├── sampling.py         # Spatial configuration sampling
│   └── visualization.py    # Plotting functions
├── notebooks/
│   └── analysis.ipynb      # Interactive analysis notebook
├── results/
│   ├── figures/           # Generated plots
│   └── data/              # Cached correlators (not in git)
├── pyproject.toml
└── README.md
```

## Performance Notes

- **Computation time**: O(N_images × N_configs × N_samples)
  - Full MNIST (60k images) with default settings: ~5-10 minutes for overall analysis
  - Per-digit analysis: ~10× longer (10 separate computations)
- **Memory**: Store correlators as float32; full C₃ can be 100s of MB
- **Speedup tips**:
  - Reduce `n_samples_per_config` and `n_configurations` for faster exploration
  - Use subset of MNIST during development
  - Results are cached in `results/data/` (add to .gitignore)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mnist_polyspectra,
  title = {MNIST N-Point Correlators and Polyspectra},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/mnist-polyspectra}
}
```

## Related Work

- **Integrated Information Theory**: Polyspectra as measures of higher-order structure
- **Turbulence/Cosmology**: Bispectrum for non-Gaussian structure in natural phenomena
- **Computer Vision**: Higher-order statistics for texture analysis
- **Spectral Learning**: Connection to Hankel matrices and WFAs

## Future Extensions

- [ ] Learned representations from trained models (hidden layer activations)
- [ ] N=4 correlators (trispectrum)
- [ ] Other datasets (CIFAR-10, natural images, textures)
- [ ] Connection to integrated information (I_N) measures
- [ ] Parallelization for faster per-digit computation
- [ ] GPU acceleration for large-scale analysis

## License

MIT License (modify as needed)

## Contact

For questions or contributions, please open an issue or submit a pull request.
