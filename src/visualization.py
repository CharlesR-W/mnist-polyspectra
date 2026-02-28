"""Visualization functions for correlators and polyspectra."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pathlib import Path

from .config import VisualizationConfig
from .polyspectra import fit_power_law


def plot_power_spectrum(
    P_data: dict,
    fit_data: dict | None = None,
    digit: int | None = None,
    ax: plt.Axes | None = None,
    config: VisualizationConfig | None = None
) -> Figure:
    """Plot radially-averaged power spectrum P(k).

    Args:
        P_data: Power spectrum data from compute_power_spectrum
        fit_data: Optional power-law fit data from fit_power_law
        digit: Digit label for title (optional)
        ax: Matplotlib axes (creates new figure if None)
        config: Visualization configuration

    Returns:
        Figure object
    """
    if config is None:
        config = VisualizationConfig()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=config.figure_dpi)
    else:
        fig = ax.figure

    k_values = P_data['k_values']
    P_k = P_data['P_k_radial']

    # Remove zeros for log-log plot
    valid = (k_values > 0) & (P_k > 0)
    k_valid = k_values[valid]
    P_valid = P_k[valid]

    # Plot data
    ax.loglog(k_valid, P_valid, 'o-', label='Data', alpha=0.7, markersize=4)

    # Plot fit if provided
    if fit_data is not None and not np.isnan(fit_data['exponent']):
        k_fit = np.logspace(np.log10(k_valid.min()), np.log10(k_valid.max()), 100)
        P_fit = fit_data['amplitude'] * k_fit**fit_data['exponent']
        ax.loglog(
            k_fit, P_fit, '--',
            label=f"$P(k) \\sim k^{{{fit_data['exponent']:.2f}}}$",
            linewidth=2
        )

    ax.set_xlabel('k (1/pixels)', fontsize=12)
    ax.set_ylabel('P(k)', fontsize=12)
    title = f'Power Spectrum'
    if digit is not None:
        title += f' - Digit {digit}'
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_power_spectrum_comparison(
    per_digit_data: dict,
    config: VisualizationConfig | None = None,
    save_path: str | None = None
) -> Figure:
    """Plot power spectra for all digits in a grid.

    Args:
        per_digit_data: Dictionary mapping digit to P_data
        config: Visualization configuration
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    if config is None:
        config = VisualizationConfig()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=config.figure_dpi)
    axes = axes.flatten()

    for digit, ax in enumerate(axes):
        if digit in per_digit_data:
            P_data = per_digit_data[digit]
            fit_data = fit_power_law(P_data['k_values'], P_data['P_k_radial'])
            plot_power_spectrum(P_data, fit_data, digit=digit, ax=ax, config=config)
        else:
            ax.axis('off')

    fig.suptitle('Power Spectra by Digit Class', fontsize=16, y=1.00)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.figure_dpi, bbox_inches='tight')

    return fig


def plot_bispectrum_equilateral(
    B_data: dict,
    digit: int | None = None,
    ax: plt.Axes | None = None,
    config: VisualizationConfig | None = None
) -> Figure:
    """Plot equilateral bispectrum configuration.

    Args:
        B_data: Bispectrum data from compute_bispectrum
        digit: Digit label for title
        ax: Matplotlib axes
        config: Visualization configuration

    Returns:
        Figure object
    """
    if config is None:
        config = VisualizationConfig()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=config.figure_dpi)
    else:
        fig = ax.figure

    if 'B_equilateral' not in B_data:
        ax.text(0.5, 0.5, 'No equilateral data', ha='center', va='center')
        return fig

    equilateral = B_data['B_equilateral']
    scales = equilateral['scales']
    values = equilateral['values']

    # Bin by scale
    scale_bins = np.linspace(scales.min(), scales.max(), 20)
    scale_centers = (scale_bins[:-1] + scale_bins[1:]) / 2
    binned_values = []
    binned_errors = []

    for i in range(len(scale_bins) - 1):
        mask = (scales >= scale_bins[i]) & (scales < scale_bins[i+1])
        if np.any(mask):
            binned_values.append(np.mean(values[mask]))
            binned_errors.append(np.std(values[mask]) / np.sqrt(np.sum(mask)))
        else:
            binned_values.append(np.nan)
            binned_errors.append(np.nan)

    binned_values = np.array(binned_values)
    binned_errors = np.array(binned_errors)

    valid = ~np.isnan(binned_values)
    ax.errorbar(
        scale_centers[valid], binned_values[valid], yerr=binned_errors[valid],
        fmt='o-', capsize=3, alpha=0.7
    )

    ax.set_xlabel('Scale (pixels)', fontsize=12)
    ax.set_ylabel('$B_{equil}$', fontsize=12)
    title = 'Equilateral Bispectrum'
    if digit is not None:
        title += f' - Digit {digit}'
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    fig.tight_layout()
    return fig


def plot_bispectrum_squeezed_heatmap(
    B_data: dict,
    digit: int | None = None,
    ax: plt.Axes | None = None,
    config: VisualizationConfig | None = None
) -> Figure:
    """Plot squeezed bispectrum configuration as heatmap.

    Args:
        B_data: Bispectrum data from compute_bispectrum
        digit: Digit label for title
        ax: Matplotlib axes
        config: Visualization configuration

    Returns:
        Figure object
    """
    if config is None:
        config = VisualizationConfig()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=config.figure_dpi)
    else:
        fig = ax.figure

    if 'B_squeezed' not in B_data:
        ax.text(0.5, 0.5, 'No squeezed data', ha='center', va='center')
        return fig

    squeezed = B_data['B_squeezed']
    r1_mag = squeezed['r1_mag']
    r2_mag = squeezed['r2_mag']
    values = squeezed['values']

    # Create 2D histogram
    bins = 20
    H, xedges, yedges = np.histogram2d(
        r1_mag, r2_mag, bins=bins, weights=values
    )
    counts, _, _ = np.histogram2d(r1_mag, r2_mag, bins=bins)
    H = np.divide(H, counts, where=counts > 0, out=np.zeros_like(H))

    # Plot
    im = ax.imshow(
        H.T, origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect='auto', cmap=config.colormap_bispectrum,
        vmin=-np.abs(H).max(), vmax=np.abs(H).max()
    )
    plt.colorbar(im, ax=ax, label='$B_{squeezed}$')

    ax.set_xlabel('$|r_1|$ (pixels)', fontsize=12)
    ax.set_ylabel('$|r_2|$ (pixels)', fontsize=12)
    title = 'Squeezed Bispectrum'
    if digit is not None:
        title += f' - Digit {digit}'
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig


def plot_bispectrum_2d(
    B_data: dict,
    digit: int | None = None,
    ax: plt.Axes | None = None,
    config: VisualizationConfig | None = None
) -> Figure:
    """Plot full 2D bispectrum B(|r1|, |r2|).

    Args:
        B_data: Bispectrum data from compute_bispectrum
        digit: Digit label for title
        ax: Matplotlib axes
        config: Visualization configuration

    Returns:
        Figure object
    """
    if config is None:
        config = VisualizationConfig()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=config.figure_dpi)
    else:
        fig = ax.figure

    if 'B_2d' not in B_data:
        ax.text(0.5, 0.5, 'No 2D bispectrum data', ha='center', va='center')
        return fig

    B_2d_data = B_data['B_2d']
    values = B_2d_data['values']
    r1_grid = B_2d_data['r1_grid']
    r2_grid = B_2d_data['r2_grid']

    im = ax.imshow(
        values, origin='lower',
        extent=[r1_grid[0], r1_grid[-1], r2_grid[0], r2_grid[-1]],
        aspect='auto', cmap=config.colormap_bispectrum,
        vmin=-np.abs(values).max(), vmax=np.abs(values).max()
    )
    plt.colorbar(im, ax=ax, label='$B(|r_1|, |r_2|)$')

    ax.set_xlabel('$|r_1|$ (pixels)', fontsize=12)
    ax.set_ylabel('$|r_2|$ (pixels)', fontsize=12)
    title = '2D Bispectrum'
    if digit is not None:
        title += f' - Digit {digit}'
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig


def plot_wavelet_spectra(
    wavelet_data: dict,
    digit: int | None = None,
    config: VisualizationConfig | None = None,
    save_path: str | None = None
) -> Figure:
    """Plot wavelet scale-dependent power spectrum.

    Args:
        wavelet_data: Wavelet spectrum data from compute_wavelet_spectrum
        digit: Digit label for title
        config: Visualization configuration
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    if config is None:
        config = VisualizationConfig()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=config.figure_dpi)

    scales = wavelet_data['scales']
    P_scale = wavelet_data['P_scale']

    ax.semilogy(scales, P_scale, 'o-', markersize=6)
    ax.set_xlabel('Wavelet Scale (pixels)', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    title = f'Wavelet Power Spectrum ({wavelet_data["wavelet"]})'
    if digit is not None:
        title += f' - Digit {digit}'
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.figure_dpi, bbox_inches='tight')

    return fig


def plot_digit_comparison_heatmap(
    all_digits_data: dict,
    config: VisualizationConfig | None = None,
    save_path: str | None = None
) -> Figure:
    """Plot heatmap comparing spectral measures across digit classes.

    Args:
        all_digits_data: Dictionary mapping digit to {'exponent', 'bispectrum_amp', ...}
        config: Visualization configuration
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    if config is None:
        config = VisualizationConfig()

    digits = sorted(all_digits_data.keys())
    metrics = ['exponent', 'bispectrum_mean', 'wavelet_scale_mean']

    # Build data matrix
    data_matrix = np.zeros((len(digits), len(metrics)))

    for i, digit in enumerate(digits):
        digit_data = all_digits_data[digit]
        data_matrix[i, 0] = digit_data.get('exponent', np.nan)
        data_matrix[i, 1] = digit_data.get('bispectrum_mean', np.nan)
        data_matrix[i, 2] = digit_data.get('wavelet_scale_mean', np.nan)

    # Normalize each column for visualization
    data_normalized = np.zeros_like(data_matrix)
    for j in range(len(metrics)):
        col = data_matrix[:, j]
        valid = ~np.isnan(col)
        if np.any(valid):
            col_min, col_max = col[valid].min(), col[valid].max()
            if col_max > col_min:
                data_normalized[:, j] = (col - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=config.figure_dpi)

    sns.heatmap(
        data_normalized,
        xticklabels=['Power Law α', 'Bispectrum Mean', 'Wavelet Scale'],
        yticklabels=[f'Digit {d}' for d in digits],
        cmap='viridis', annot=True, fmt='.2f',
        cbar_kws={'label': 'Normalized Value'},
        ax=ax
    )

    ax.set_title('Spectral Measures by Digit Class', fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.figure_dpi, bbox_inches='tight')

    return fig


def create_summary_figure(
    P_data: dict,
    B_data: dict,
    wavelet_data: dict | None = None,
    fit_data: dict | None = None,
    digit: int | None = None,
    config: VisualizationConfig | None = None,
    save_path: str | None = None
) -> Figure:
    """Create multi-panel summary figure with all spectral measures.

    Args:
        P_data: Power spectrum data
        B_data: Bispectrum data
        wavelet_data: Wavelet spectrum data (optional)
        fit_data: Power-law fit data (optional)
        digit: Digit label for title
        config: Visualization configuration
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    if config is None:
        config = VisualizationConfig()

    if wavelet_data is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=config.figure_dpi)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=config.figure_dpi)

    axes = axes.flatten()

    # Power spectrum
    plot_power_spectrum(P_data, fit_data, digit, ax=axes[0], config=config)

    # Equilateral bispectrum
    plot_bispectrum_equilateral(B_data, digit, ax=axes[1], config=config)

    # 2D bispectrum
    plot_bispectrum_2d(B_data, digit, ax=axes[2], config=config)

    # Squeezed bispectrum
    plot_bispectrum_squeezed_heatmap(B_data, digit, ax=axes[3], config=config)

    # Wavelet spectrum (if available)
    if wavelet_data is not None and len(axes) > 4:
        scales = wavelet_data['scales']
        P_scale = wavelet_data['P_scale']
        axes[4].semilogy(scales, P_scale, 'o-', markersize=6)
        axes[4].set_xlabel('Wavelet Scale (pixels)', fontsize=12)
        axes[4].set_ylabel('Power', fontsize=12)
        axes[4].set_title(f'Wavelet Spectrum ({wavelet_data["wavelet"]})', fontsize=12)
        axes[4].grid(True, alpha=0.3)

        # Hide last panel
        axes[5].axis('off')

    suptitle = 'MNIST Polyspectra Summary'
    if digit is not None:
        suptitle += f' - Digit {digit}'
    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.figure_dpi, bbox_inches='tight')

    return fig
