"""Configuration dataclasses for correlator and polyspectra computation."""

from dataclasses import dataclass, field


@dataclass
class CorrelatorConfig:
    """Configuration for N-point correlator computation.

    Attributes:
        n_points: Number of points in correlator (2 or 3)
        max_separation: Maximum separation distance in pixels
        n_samples_per_config: Number of anchor points to sample per configuration
        n_configurations: Number of separation configurations to sample
        translation_invariant: Whether to assume translation invariance
    """
    n_points: int = 3
    max_separation: int = 14
    n_samples_per_config: int = 1000
    n_configurations: int = 500
    translation_invariant: bool = True


@dataclass
class SamplingConfig:
    """Configuration for spatial configuration sampling.

    Attributes:
        separation_grid: Grid of separation magnitudes to sample (pixels)
        n_angles: Number of angular samples for each separation magnitude
        emphasize_geometries: Geometric configurations to oversample
        geometry_oversample: Oversampling factor for emphasized geometries
    """
    separation_grid: list[int] = field(default_factory=lambda: [1, 2, 4, 7, 10, 14])
    n_angles: int = 8
    emphasize_geometries: list[str] = field(
        default_factory=lambda: ['equilateral', 'squeezed', 'collinear']
    )
    geometry_oversample: float = 2.0


@dataclass
class PolyspectraConfig:
    """Configuration for polyspectra computation.

    Attributes:
        use_fourier: Whether to compute Fourier polyspectra
        use_wavelet: Whether to compute wavelet polyspectra
        wavelet_family: Wavelet family for DWT (e.g., 'db4', 'sym4')
        n_wavelet_scales: Number of wavelet decomposition scales
        bispectrum_slices: Types of bispectrum slices to extract
    """
    use_fourier: bool = True
    use_wavelet: bool = True
    wavelet_family: str = 'db4'
    n_wavelet_scales: int = 3
    bispectrum_slices: list[str] = field(
        default_factory=lambda: ['equilateral', 'squeezed', 'folded']
    )


@dataclass
class VisualizationConfig:
    """Configuration for visualization and plotting.

    Attributes:
        per_digit: Whether to create per-digit visualizations
        figure_dpi: DPI for saved figures
        save_format: Image format for saved figures
        colormap_bispectrum: Colormap for bispectrum plots (diverging)
    """
    per_digit: bool = True
    figure_dpi: int = 150
    save_format: str = 'png'
    colormap_bispectrum: str = 'RdBu_r'
