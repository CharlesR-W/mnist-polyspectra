"""Polyspectra computation via Fourier and wavelet transforms."""

import numpy as np
import pywt
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.interpolate import griddata

from .config import PolyspectraConfig


def compute_power_spectrum(
    C2_data: dict,
    config: PolyspectraConfig | None = None
) -> dict:
    """Compute power spectrum from 2-point correlator.

    P(k) = FFT[C₂(r)]

    Args:
        C2_data: Dictionary from compute_connected_2point
        config: Polyspectra configuration

    Returns:
        Dictionary containing:
            - 'P_k_2d': 2D power spectrum in k-space
            - 'P_k_radial': Radially-averaged power spectrum
            - 'k_values': k-magnitude values for radial average
            - 'k_x': k_x grid
            - 'k_y': k_y grid
    """
    if config is None:
        config = PolyspectraConfig()

    # Interpolate C2 onto regular grid
    separations = C2_data['separations']
    C2_values = C2_data['C2']

    # Remove NaN values
    valid = ~np.isnan(C2_values)
    separations = separations[valid]
    C2_values = C2_values[valid]

    # Create regular grid
    max_r = np.max(np.abs(separations))
    grid_size = 64  # FFT works best with power of 2
    x_grid = np.linspace(-max_r, max_r, grid_size)
    y_grid = np.linspace(-max_r, max_r, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Interpolate C2 onto grid
    C2_grid = griddata(
        separations, C2_values, (X, Y),
        method='linear', fill_value=0.0
    )

    # Compute 2D FFT
    P_k_2d = np.abs(np.fft.fftshift(np.fft.fft2(C2_grid)))**2
    k_x = np.fft.fftshift(np.fft.fftfreq(grid_size, d=(x_grid[1] - x_grid[0])))
    k_y = np.fft.fftshift(np.fft.fftfreq(grid_size, d=(y_grid[1] - y_grid[0])))

    # Radial average
    K_x, K_y = np.meshgrid(k_x, k_y)
    K_mag = np.sqrt(K_x**2 + K_y**2)

    k_bins = np.linspace(0, np.max(K_mag), 30)
    k_values = (k_bins[:-1] + k_bins[1:]) / 2
    P_k_radial = np.zeros(len(k_values))

    for i, k in enumerate(k_values):
        mask = (K_mag >= k_bins[i]) & (K_mag < k_bins[i+1])
        if np.any(mask):
            P_k_radial[i] = np.mean(P_k_2d[mask])

    return {
        'P_k_2d': P_k_2d.astype(np.float32),
        'P_k_radial': P_k_radial.astype(np.float32),
        'k_values': k_values.astype(np.float32),
        'k_x': k_x.astype(np.float32),
        'k_y': k_y.astype(np.float32)
    }


def compute_bispectrum(
    C3_data: dict,
    config: PolyspectraConfig | None = None
) -> dict:
    """Compute bispectrum from 3-point correlator.

    B(k₁, k₂, k₃) = FFT[C₃(r₁, r₂)] with k₁ + k₂ + k₃ = 0

    Args:
        C3_data: Dictionary from compute_connected_3point
        config: Polyspectra configuration

    Returns:
        Dictionary containing bispectrum slices and full 2D transform
    """
    if config is None:
        config = PolyspectraConfig()

    r1_vectors = C3_data['r1_vectors']
    r2_vectors = C3_data['r2_vectors']
    C3_values = C3_data['C3']

    # Remove NaN values
    valid = ~np.isnan(C3_values)
    r1_vectors = r1_vectors[valid]
    r2_vectors = r2_vectors[valid]
    C3_values = C3_values[valid]

    # Create 4D grid for (r1_x, r1_y, r2_x, r2_y)
    # For computational efficiency, we'll project to specific slices

    results = {}

    # Equilateral configuration: |k1| = |k2| = |k3|
    # Extract equilateral triplets from r1, r2
    r1_mag = np.linalg.norm(r1_vectors, axis=1)
    r2_mag = np.linalg.norm(r2_vectors, axis=1)
    r12_diff = np.linalg.norm(r2_vectors - r1_vectors, axis=1)

    # Find approximately equilateral configurations
    equilateral_tol = 0.3  # 30% tolerance
    equilateral_mask = (
        np.abs(r1_mag - r2_mag) / (r1_mag + 1e-6) < equilateral_tol
    ) & (
        np.abs(r1_mag - r12_diff) / (r1_mag + 1e-6) < equilateral_tol
    )

    if np.any(equilateral_mask):
        equilateral_C3 = C3_values[equilateral_mask]
        equilateral_scales = r1_mag[equilateral_mask]
        results['B_equilateral'] = {
            'values': equilateral_C3.astype(np.float32),
            'scales': equilateral_scales.astype(np.float32)
        }

    # Squeezed configuration: r1 ≈ r2 (small angle)
    angle_diff = np.arccos(
        np.clip(
            np.sum(r1_vectors * r2_vectors, axis=1) / (r1_mag * r2_mag + 1e-6),
            -1, 1
        )
    )
    squeezed_mask = angle_diff < np.pi / 6  # < 30 degrees

    if np.any(squeezed_mask):
        squeezed_C3 = C3_values[squeezed_mask]
        squeezed_r1 = r1_mag[squeezed_mask]
        squeezed_r2 = r2_mag[squeezed_mask]
        results['B_squeezed'] = {
            'values': squeezed_C3.astype(np.float32),
            'r1_mag': squeezed_r1.astype(np.float32),
            'r2_mag': squeezed_r2.astype(np.float32)
        }

    # Full 2D bispectrum in (r1, r2) plane (simplified projection)
    # Grid r1 and r2 magnitudes
    max_r = max(np.max(r1_mag), np.max(r2_mag))
    r_grid = np.linspace(0, max_r, 32)
    R1, R2 = np.meshgrid(r_grid, r_grid)

    # Average C3 over angles for each (|r1|, |r2|)
    B_2d = np.zeros((32, 32))
    counts = np.zeros((32, 32))

    for i in range(len(C3_values)):
        r1_idx = np.argmin(np.abs(r_grid - r1_mag[i]))
        r2_idx = np.argmin(np.abs(r_grid - r2_mag[i]))
        B_2d[r2_idx, r1_idx] += C3_values[i]
        counts[r2_idx, r1_idx] += 1

    B_2d = np.divide(B_2d, counts, where=counts > 0, out=np.zeros_like(B_2d))

    results['B_2d'] = {
        'values': B_2d.astype(np.float32),
        'r1_grid': r_grid.astype(np.float32),
        'r2_grid': r_grid.astype(np.float32)
    }

    return results


def compute_wavelet_spectrum(
    C2_data: dict,
    config: PolyspectraConfig | None = None
) -> dict:
    """Compute wavelet-based power spectrum from C2.

    Args:
        C2_data: Dictionary from compute_connected_2point
        config: Polyspectra configuration

    Returns:
        Dictionary containing scale-dependent power spectrum
    """
    if config is None:
        config = PolyspectraConfig()

    # Interpolate C2 onto regular grid
    separations = C2_data['separations']
    C2_values = C2_data['C2']

    valid = ~np.isnan(C2_values)
    separations = separations[valid]
    C2_values = C2_values[valid]

    max_r = np.max(np.abs(separations))
    grid_size = 64
    x_grid = np.linspace(-max_r, max_r, grid_size)
    y_grid = np.linspace(-max_r, max_r, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    C2_grid = griddata(
        separations, C2_values, (X, Y),
        method='linear', fill_value=0.0
    )

    # 2D wavelet decomposition
    coeffs = pywt.wavedec2(C2_grid, config.wavelet_family, level=config.n_wavelet_scales)

    # Extract power at each scale
    scale_powers = []
    scales = []

    for i, coeff_level in enumerate(coeffs):
        if i == 0:
            # Approximation coefficients
            power = np.mean(np.abs(coeff_level)**2)
            scale = 2**config.n_wavelet_scales
        else:
            # Detail coefficients (cH, cV, cD)
            cH, cV, cD = coeff_level
            power = np.mean(np.abs(cH)**2 + np.abs(cV)**2 + np.abs(cD)**2)
            scale = 2**(config.n_wavelet_scales - i + 1)

        scale_powers.append(power)
        scales.append(scale)

    return {
        'P_scale': np.array(scale_powers, dtype=np.float32),
        'scales': np.array(scales, dtype=np.float32),
        'wavelet': config.wavelet_family
    }


def fit_power_law(k_values: NDArray, P_k: NDArray) -> dict:
    """Fit power-law P(k) ~ A k^α to power spectrum.

    Args:
        k_values: k-magnitude values
        P_k: Power spectrum values

    Returns:
        Dictionary with 'exponent', 'amplitude', 'r_squared'
    """
    # Remove zero and negative values
    valid = (k_values > 0) & (P_k > 0)
    if np.sum(valid) < 3:
        return {
            'exponent': np.nan,
            'amplitude': np.nan,
            'r_squared': np.nan
        }

    k_valid = k_values[valid]
    P_valid = P_k[valid]

    # Fit in log-log space
    log_k = np.log(k_valid)
    log_P = np.log(P_valid)

    def power_law_log(log_k, log_A, alpha):
        return log_A + alpha * log_k

    try:
        popt, _ = curve_fit(power_law_log, log_k, log_P)
        log_A, alpha = popt

        # Compute R²
        P_fit = np.exp(log_A) * k_valid**alpha
        residuals = log_P - np.log(P_fit)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_P - np.mean(log_P))**2)
        r_squared = 1 - ss_res / ss_tot

        return {
            'exponent': float(alpha),
            'amplitude': float(np.exp(log_A)),
            'r_squared': float(r_squared)
        }
    except:
        return {
            'exponent': np.nan,
            'amplitude': np.nan,
            'r_squared': np.nan
        }
