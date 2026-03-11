# MNIST Polyspectra

**Companion code for [Power Laws Are Not Enough](https://crw.dev/posts/Power-Laws-Are-Not-Enough/) and [The Spectral Structure of Natural Data](https://crw.dev/posts/The-Spectral-Structure-of-Natural-Data/).**

Computes N-point connected correlators (cumulants) on MNIST images and transforms them to Fourier/wavelet polyspectra, measuring higher-order statistical structure beyond the power spectrum.

> **Research companion code.**  This repository implements the correlator computations and spectral analyses discussed in the blog posts on data geometry and scaling laws.  It is a work in progress - functional and producing results, but should be understood as research infrastructure rather than a polished library.  Developed during [MATS](https://www.matsprogram.org/) 9.0.

## What it computes

**2-point correlators** $C_2(r) = \langle I(x) I(x+r) \rangle - \langle I \rangle^2$, transformed to the **power spectrum** $P(k)$ via FFT.  Natural images show power-law decay $P(k) \sim k^\alpha$ with $\alpha \approx -2$ (scale invariance).

**3-point correlators** $C_3(r_1, r_2)$ (connected cumulants), transformed to the **bispectrum** $B(k_1, k_2, k_3)$ in three canonical slices (equilateral, squeezed, folded).  Non-zero bispectrum indicates non-Gaussian, nonlinear structure - the "beyond power law" information that distinguishes natural images from Gaussian random fields with the same spectrum.

**Per-digit analysis** computes correlators separately for each digit class (0-9) to detect statistical differences across categories.

This is a **computational experiment**: it loads MNIST, computes correlators from scratch via Monte Carlo sampling, and caches results.  The marimo notebook provides interactive visualization of the computed results.

## Quick start

```bash
uv sync

# Run the interactive analysis notebook
marimo run notebooks/analysis.py --sandbox

# Or from Python
python -c "
from src.correlators import compute_connected_2point
from src.polyspectra import compute_power_spectrum, fit_power_law
from src.config import CorrelatorConfig, SamplingConfig
from sklearn.datasets import fetch_openml

images = fetch_openml('mnist_784', version=1, parser='auto').data.values.reshape(-1, 28, 28)[:1000]
C2 = compute_connected_2point(images, CorrelatorConfig(), SamplingConfig())
P = compute_power_spectrum(C2)
fit = fit_power_law(P['k_values'], P['P_k_radial'])
print(f'Power-law exponent: alpha = {fit[\"exponent\"]:.3f}')
"
```

## Project structure

```
src/
├── config.py           # Configuration dataclasses
├── correlators.py      # N-point correlator computation
├── polyspectra.py      # Fourier/wavelet transforms + power-law fitting
├── sampling.py         # Spatial configuration sampling
└── visualization.py    # Plotting functions
notebooks/
├── analysis.py         # Marimo notebook (primary interface)
└── analysis.ipynb      # Jupyter version
results/
├── figures/            # Generated plots
└── data/               # Cached correlators (.npz, not in git)
```

## Related posts

- [Power Laws Are Not Enough](https://crw.dev/posts/Power-Laws-Are-Not-Enough/) - why covariance spectra don't tell the whole story
- [The Spectral Structure of Natural Data](https://crw.dev/posts/The-Spectral-Structure-of-Natural-Data/) - the data-side geometry that polyspectra characterize
- [The Lamppost Hypothesis](https://crw.dev/posts/The-Lamppost-Hypothesis/) - why data has cooperative spectral structure

## License

MIT

---

*Developed during MATS 9.0.  Written with Claude.*
