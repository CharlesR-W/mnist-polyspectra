# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "scipy>=1.10",
#     "PyWavelets>=1.4",
#     "scikit-learn>=1.3",
#     "matplotlib>=3.7",
#     "seaborn>=0.12",
#     "tqdm>=4.65",
#     "marimo",
#     "mnist-polyspectra",
# ]
#
# [tool.marimo]
# width = "medium"
# theme = "dark"
# ///

import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# MNIST N-Point Correlators & Polyspectra

    Compute N=2,3 connected correlators (cumulants) on MNIST images and transform
    them to Fourier/wavelet polyspectra.

    - **2-point**: $C_2(r) = \langle I(x)\, I(x+r) \rangle - \langle I \rangle^2 \;\to\; P(k)$
    - **3-point**: $C_3(r_1, r_2)$ (cumulant) $\;\to\;$ bispectrum slices
    - Per-digit class comparisons across 0-9
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_samples_slider = mo.ui.slider(
        start=1000, stop=20000, step=1000, value=10000,
        label="Number of MNIST images"
    )
    n_configs_slider = mo.ui.slider(
        start=100, stop=2000, step=100, value=500,
        label="C₃ configurations (overall)"
    )
    n_samples_per_config = mo.ui.slider(
        start=100, stop=2000, step=100, value=500,
        label="Anchor points per config"
    )
    mo.md(f"""## Configuration

    {n_samples_slider}

    {n_configs_slider}

    {n_samples_per_config}
    """)
    return n_configs_slider, n_samples_per_config, n_samples_slider


@app.cell
def _(n_configs_slider, n_samples_per_config, n_samples_slider, np):
    from sklearn.datasets import fetch_openml
    from src.config import CorrelatorConfig, SamplingConfig, PolyspectraConfig, VisualizationConfig

    # Load MNIST
    _mnist = fetch_openml('mnist_784', version=1, parser='auto')
    _all_images = _mnist.data.values.reshape(-1, 28, 28).astype(np.float32)
    _all_labels = _mnist.target.astype(int).values

    _rng = np.random.RandomState(42)
    _indices = _rng.choice(len(_all_images), n_samples_slider.value, replace=False)
    images = _all_images[_indices]
    labels = _all_labels[_indices]

    corr_config = CorrelatorConfig(
        n_points=3,
        max_separation=14,
        n_samples_per_config=n_samples_per_config.value,
        n_configurations=n_configs_slider.value,
        translation_invariant=True,
    )
    samp_config = SamplingConfig(
        separation_grid=[1, 2, 4, 7, 10, 14],
        n_angles=12,
        emphasize_geometries=['equilateral', 'squeezed', 'collinear'],
        geometry_oversample=3.0,
    )
    poly_config = PolyspectraConfig()
    vis_config = VisualizationConfig(per_digit=True, figure_dpi=150)

    print(f"Loaded {len(images)} images, shape {images.shape[1:]}")
    return (
        PolyspectraConfig,
        VisualizationConfig,
        corr_config,
        images,
        labels,
        poly_config,
        samp_config,
        vis_config,
    )


@app.cell(hide_code=True)
def _(images, labels, mo, np, plt):
    _fig, _axes = plt.subplots(2, 5, figsize=(12, 5))
    _fig.patch.set_facecolor('#1a1a2e')
    for _digit in range(10):
        _idx = np.where(labels == _digit)[0][0]
        _ax = _axes[_digit // 5, _digit % 5]
        _ax.imshow(images[_idx], cmap='gray')
        _ax.set_title(f'Digit {_digit}', color='white', fontsize=11)
        _ax.axis('off')
    _fig.suptitle('Sample MNIST Images', color='white', fontsize=14)
    _fig.tight_layout()
    mo.md("## Sample Images")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Overall Correlators

    Computing $C_2(r)$ and $C_3(r_1, r_2)$ across all images.
    Images are centered ($\langle I \rangle = 0$) so the connected correlators simplify:
    - $C_2(r) = \langle I(x)\, I(x+r) \rangle$
    - $C_3(r_1, r_2) = \langle I(x)\, I(x+r_1)\, I(x+r_2) \rangle$
    """)
    return


@app.cell
def _(corr_config, images, samp_config):
    from src.correlators import compute_connected_2point, compute_connected_3point

    C2_result = compute_connected_2point(images, corr_config, samp_config, verbose=True)
    C3_result = compute_connected_3point(images, corr_config, samp_config, verbose=True)

    print(f"C2: {len(C2_result['C2'])} configs, C3: {len(C3_result['C3'])} configs")
    return C2_result, C3_result


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Fourier Polyspectra")
    return


@app.cell
def _(C2_result, poly_config):
    from src.polyspectra import compute_power_spectrum, fit_power_law

    P_data = compute_power_spectrum(C2_result, poly_config)
    fit_data = fit_power_law(P_data['k_values'], P_data['P_k_radial'])
    print(f"Power-law exponent: α = {fit_data['exponent']:.3f} (R² = {fit_data['r_squared']:.3f})")
    return P_data, fit_data


@app.cell(hide_code=True)
def _(P_data, fit_data, mo, np, plt):
    _fig, _ax = plt.subplots(figsize=(9, 6))
    _fig.patch.set_facecolor('#1a1a2e')
    _ax.set_facecolor('#1a1a2e')

    _k = P_data['k_values']
    _P = P_data['P_k_radial']
    _valid = (_k > 0) & (_P > 0)

    _ax.loglog(_k[_valid], _P[_valid], 'o-', color='#7dd3fc', alpha=0.8, markersize=4, label='Data')

    if not np.isnan(fit_data['exponent']):
        _k_fit = np.logspace(np.log10(_k[_valid].min()), np.log10(_k[_valid].max()), 100)
        _P_fit = fit_data['amplitude'] * _k_fit**fit_data['exponent']
        _ax.loglog(_k_fit, _P_fit, '--', color='#fcd34d', linewidth=2,
                   label=f"$P(k) \\sim k^{{{fit_data['exponent']:.2f}}}$")

    _ax.set_xlabel('k (1/pixels)', fontsize=12, color='white')
    _ax.set_ylabel('P(k)', fontsize=12, color='white')
    _ax.set_title('Power Spectrum (Wiener-Khinchin)', fontsize=14, color='white')
    _ax.legend(fontsize=11)
    _ax.grid(True, alpha=0.2)
    _ax.tick_params(colors='#94a3b8')
    _fig.tight_layout()

    mo.md(f"""### Power Spectrum

    $P(k) = \\text{{FT}}[C_2(r)]$ with power-law fit $P(k) \\sim k^{{{fit_data['exponent']:.2f}}}$
    ($R^2 = {fit_data['r_squared']:.3f}$).

    For reference, natural images typically show $\\alpha \\approx -2$.
    MNIST digits have more concentrated spatial structure (sharp edges, thin strokes)
    leading to a shallower falloff.
    """)
    return


@app.cell
def _(C3_result, poly_config):
    from src.polyspectra import compute_bispectrum

    B_data = compute_bispectrum(C3_result, poly_config)
    _n_eq = len(B_data.get('B_equilateral', {}).get('values', []))
    _n_sq = len(B_data.get('B_squeezed', {}).get('values', []))
    print(f"Bispectrum: {_n_eq} equilateral, {_n_sq} squeezed configurations")
    return (B_data,)


@app.cell(hide_code=True)
def _(B_data, mo, np, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 6))
    _fig.patch.set_facecolor('#1a1a2e')

    # Equilateral bispectrum
    _ax1.set_facecolor('#1a1a2e')
    if 'B_equilateral' in B_data:
        _eq = B_data['B_equilateral']
        _scales = _eq['scales']
        _values = _eq['values']

        _bins = np.linspace(_scales.min(), _scales.max(), 12)
        _centers = (_bins[:-1] + _bins[1:]) / 2
        _binned = []
        _errors = []
        for _i in range(len(_bins) - 1):
            _mask = (_scales >= _bins[_i]) & (_scales < _bins[_i + 1])
            if np.any(_mask):
                _binned.append(np.mean(_values[_mask]))
                _errors.append(np.std(_values[_mask]) / np.sqrt(np.sum(_mask)))
            else:
                _binned.append(np.nan)
                _errors.append(np.nan)
        _binned = np.array(_binned)
        _errors = np.array(_errors)
        _v = ~np.isnan(_binned)
        _ax1.errorbar(_centers[_v], _binned[_v], yerr=_errors[_v],
                      fmt='o-', color='#6ee7b7', capsize=3, alpha=0.8)
    _ax1.axhline(0, color='#94a3b8', linestyle='--', alpha=0.4)
    _ax1.set_xlabel('Scale (pixels)', fontsize=12, color='white')
    _ax1.set_ylabel('$B_{equil}$', fontsize=12, color='white')
    _ax1.set_title('Equilateral Bispectrum', fontsize=14, color='white')
    _ax1.grid(True, alpha=0.2)
    _ax1.tick_params(colors='#94a3b8')

    # 2D bispectrum
    _ax2.set_facecolor('#1a1a2e')
    if 'B_2d' in B_data:
        _b2d = B_data['B_2d']
        _vmax = np.abs(_b2d['values']).max()
        _im = _ax2.imshow(
            _b2d['values'], origin='lower',
            extent=[_b2d['r1_grid'][0], _b2d['r1_grid'][-1],
                    _b2d['r2_grid'][0], _b2d['r2_grid'][-1]],
            aspect='auto', cmap='RdBu_r', vmin=-_vmax, vmax=_vmax
        )
        plt.colorbar(_im, ax=_ax2, label='$B(|r_1|, |r_2|)$')
    _ax2.set_xlabel('$|r_1|$ (pixels)', fontsize=12, color='white')
    _ax2.set_ylabel('$|r_2|$ (pixels)', fontsize=12, color='white')
    _ax2.set_title('2D Bispectrum', fontsize=14, color='white')
    _ax2.tick_params(colors='#94a3b8')
    _fig.tight_layout()

    mo.md(r"""### Bispectrum

    Non-zero bispectrum indicates **non-Gaussian structure** in MNIST images.
    The equilateral configuration $|r_1| = |r_2| = |r_2 - r_1|$ probes
    symmetric three-point correlations at each scale.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Wavelet Polyspectra")
    return


@app.cell
def _(C2_result, poly_config):
    from src.polyspectra import compute_wavelet_spectrum

    wavelet_data = compute_wavelet_spectrum(C2_result, poly_config)
    return (wavelet_data,)


@app.cell(hide_code=True)
def _(mo, plt, wavelet_data):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _fig.patch.set_facecolor('#1a1a2e')
    _ax.set_facecolor('#1a1a2e')

    _scales = wavelet_data['scales']
    _power = wavelet_data['P_scale']

    _ax.semilogy(_scales, _power, 'o-', color='#c4b5fd', markersize=8)
    _ax.set_xlabel('Wavelet Scale (pixels)', fontsize=12, color='white')
    _ax.set_ylabel('Power', fontsize=12, color='white')
    _ax.set_title(f'Wavelet Power Spectrum ({wavelet_data["wavelet"]})', fontsize=14, color='white')
    _ax.grid(True, alpha=0.2)
    _ax.tick_params(colors='#94a3b8')
    _fig.tight_layout()

    mo.md(r"""### Scale-Dependent Power

    Wavelet decomposition of $C_2(r)$ using Daubechies-4 wavelets.
    Power increases with scale, consistent with the power spectrum -
    most variance is at large spatial scales (overall digit shape)
    rather than fine detail (edges, texture).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Per-Digit Analysis

    Computing correlators separately for each digit class (0-9) with
    reduced sampling for speed.
    """)
    return


@app.cell
def _(PolyspectraConfig, VisualizationConfig, images, labels, np, samp_config):
    from src.config import CorrelatorConfig as _CC
    from src.correlators import compute_per_digit_correlators
    from src.polyspectra import (
        compute_bispectrum as _bispectrum,
        compute_power_spectrum as _power,
        compute_wavelet_spectrum as _wavelet,
        fit_power_law as _fit,
    )

    _corr_fast = _CC(
        n_points=3,
        max_separation=14,
        n_samples_per_config=200,
        n_configurations=300,
        translation_invariant=True,
    )

    per_digit_correlators = compute_per_digit_correlators(
        images, labels, _corr_fast, samp_config, verbose=True
    )

    _poly = PolyspectraConfig()
    per_digit_P = {}
    per_digit_B = {}
    per_digit_wavelet = {}
    per_digit_fits = {}

    for _d in range(10):
        _c2 = per_digit_correlators[_d]['C2']
        _c3 = per_digit_correlators[_d]['C3']
        _pd = _power(_c2, _poly)
        _fd = _fit(_pd['k_values'], _pd['P_k_radial'])
        per_digit_P[_d] = _pd
        per_digit_fits[_d] = _fd
        per_digit_B[_d] = _bispectrum(_c3, _poly)
        per_digit_wavelet[_d] = _wavelet(_c2, _poly)
        print(f"Digit {_d}: α = {_fd['exponent']:.3f}")

    all_digits_stats = {}
    for _d in range(10):
        all_digits_stats[_d] = {
            'exponent': per_digit_fits[_d]['exponent'],
            'bispectrum_mean': np.nanmean(per_digit_B[_d].get('B_equilateral', {}).get('values', [0])),
            'wavelet_scale_mean': np.mean(per_digit_wavelet[_d]['P_scale']),
        }
    return (
        all_digits_stats,
        per_digit_B,
        per_digit_P,
        per_digit_correlators,
        per_digit_fits,
        per_digit_wavelet,
    )


@app.cell(hide_code=True)
def _(mo, np, per_digit_P, plt):
    from src.polyspectra import fit_power_law as _fit_pl

    _fig, _axes = plt.subplots(2, 5, figsize=(20, 8))
    _fig.patch.set_facecolor('#1a1a2e')

    _colors = ['#7dd3fc', '#fcd34d', '#6ee7b7', '#fca5a5', '#c4b5fd',
               '#5eead4', '#7dd3fc', '#fcd34d', '#6ee7b7', '#fca5a5']

    for _d, _ax in enumerate(_axes.flatten()):
        _ax.set_facecolor('#1a1a2e')
        if _d in per_digit_P:
            _pd = per_digit_P[_d]
            _fd = _fit_pl(_pd['k_values'], _pd['P_k_radial'])
            _k = _pd['k_values']
            _P = _pd['P_k_radial']
            _v = (_k > 0) & (_P > 0)
            _ax.loglog(_k[_v], _P[_v], 'o-', color=_colors[_d], alpha=0.7, markersize=3)
            if not np.isnan(_fd['exponent']):
                _kf = np.logspace(np.log10(_k[_v].min()), np.log10(_k[_v].max()), 50)
                _ax.loglog(_kf, _fd['amplitude'] * _kf**_fd['exponent'],
                           '--', color='#94a3b8', linewidth=1.5)
            _ax.set_title(f"Digit {_d}: α={_fd['exponent']:.2f}",
                          fontsize=10, color='white')
        _ax.grid(True, alpha=0.15)
        _ax.tick_params(colors='#94a3b8', labelsize=8)
        if _d % 5 == 0:
            _ax.set_ylabel('P(k)', fontsize=9, color='white')
        if _d >= 5:
            _ax.set_xlabel('k', fontsize=9, color='white')

    _fig.suptitle('Power Spectra by Digit Class', fontsize=14, color='white')
    _fig.tight_layout()

    mo.md("### Per-Digit Power Spectra")
    return


@app.cell(hide_code=True)
def _(all_digits_stats, mo, np, plt, sns):
    _digits = sorted(all_digits_stats.keys())
    _metrics = ['exponent', 'bispectrum_mean', 'wavelet_scale_mean']
    _labels_display = ['Power Law α', 'Bispectrum Mean', 'Wavelet Scale']

    _data = np.zeros((len(_digits), len(_metrics)))
    for _i, _d in enumerate(_digits):
        _data[_i, 0] = all_digits_stats[_d].get('exponent', np.nan)
        _data[_i, 1] = all_digits_stats[_d].get('bispectrum_mean', np.nan)
        _data[_i, 2] = all_digits_stats[_d].get('wavelet_scale_mean', np.nan)

    _norm = np.zeros_like(_data)
    for _j in range(len(_metrics)):
        _col = _data[:, _j]
        _v = ~np.isnan(_col)
        if np.any(_v):
            _mn, _mx = _col[_v].min(), _col[_v].max()
            if _mx > _mn:
                _norm[:, _j] = (_col - _mn) / (_mx - _mn)

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _fig.patch.set_facecolor('#1a1a2e')
    _ax.set_facecolor('#1a1a2e')

    sns.heatmap(
        _norm,
        xticklabels=_labels_display,
        yticklabels=[f'Digit {_d}' for _d in _digits],
        cmap='viridis', annot=True, fmt='.2f',
        cbar_kws={'label': 'Normalized Value'},
        ax=_ax,
    )
    _ax.set_title('Spectral Measures by Digit Class', fontsize=14, color='white')
    _ax.tick_params(colors='#94a3b8')
    _fig.tight_layout()

    mo.md(r"""### Digit Comparison Heatmap

    Normalized spectral measures across digit classes.  Key patterns:
    - Digits **0, 2, 3** have high bispectrum (strongest non-Gaussian structure - curved strokes)
    - Digit **1** has lowest wavelet power (thinnest, most spatially concentrated)
    - Digit **8** has highest wavelet power (most spread-out spatial structure)
    """)
    return


@app.cell(hide_code=True)
def _(all_digits_stats, mo):
    _rows = []
    for _d in range(10):
        _s = all_digits_stats[_d]
        _rows.append(
            f"| {_d} | {_s['exponent']:.3f} | {_s['bispectrum_mean']:.2e} | {_s['wavelet_scale_mean']:.2e} |"
        )

    mo.md(f"""### Summary Statistics

    | Digit | Power-law α | Bispectrum Mean | Wavelet Scale Mean |
    |:-----:|:-----------:|:---------------:|:------------------:|
    {chr(10).join(_rows)}
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    plt.style.use('dark_background')
    return mo, np, plt, sns


if __name__ == "__main__":
    app.run()
