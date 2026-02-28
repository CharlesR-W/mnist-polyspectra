"""N-point connected correlator (cumulant) computation."""

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .config import CorrelatorConfig, SamplingConfig
from .sampling import (
    compute_pair_products,
    compute_triplet_products,
    generate_separation_vectors,
    sample_anchor_points,
)


def compute_connected_2point(
    images: NDArray,
    config: CorrelatorConfig,
    sampling_config: SamplingConfig | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = True
) -> dict:
    """Compute 2-point connected correlator (cumulant).

    C₂(r) = ⟨I(x) I(x+r)⟩ - ⟨I(x)⟩²

    Args:
        images: Image batch, shape (n_images, height, width)
        config: Correlator configuration
        sampling_config: Sampling configuration (default: SamplingConfig())
        rng: Random number generator
        verbose: Whether to show progress bar

    Returns:
        Dictionary containing:
            - 'C2': Connected 2-point correlator values, shape (n_configs,)
            - 'separations': Separation vectors, shape (n_configs, 2)
            - 'errors': Standard errors, shape (n_configs,)
            - 'mean_intensity': Mean image intensity
    """
    if rng is None:
        rng = np.random.default_rng()
    if sampling_config is None:
        sampling_config = SamplingConfig()

    # Normalize images
    images = images.astype(np.float32)
    mean_intensity = np.mean(images)
    images_centered = images - mean_intensity

    image_shape = images.shape[1:]

    # Generate separation vectors
    # For 2-point, just sample single separations
    n_configs = config.n_configurations
    separations = []
    for _ in range(n_configs):
        r_mag = rng.choice(sampling_config.separation_grid)
        angle = rng.uniform(0, 2 * np.pi)
        r = np.array([r_mag * np.cos(angle), r_mag * np.sin(angle)])
        separations.append(r)
    separations = np.array(separations)

    # Compute correlators
    C2_values = []
    errors = []

    iterator = enumerate(separations)
    if verbose:
        iterator = tqdm(list(iterator), desc="Computing C2")

    for idx, r in iterator:
        # Sample anchor points
        x_anchors = sample_anchor_points(
            image_shape, r, np.zeros(2), config.n_samples_per_config, rng
        )

        if len(x_anchors) == 0:
            C2_values.append(np.nan)
            errors.append(np.nan)
            continue

        # Compute pair products
        products = compute_pair_products(images_centered, x_anchors, r)

        # Average over images and anchor points
        # C2 = ⟨I(x) I(x+r)⟩ (already centered, so no need to subtract mean²)
        C2 = np.mean(products)
        C2_error = np.std(products) / np.sqrt(products.size)

        C2_values.append(C2)
        errors.append(C2_error)

    return {
        'C2': np.array(C2_values, dtype=np.float32),
        'separations': separations.astype(np.float32),
        'errors': np.array(errors, dtype=np.float32),
        'mean_intensity': float(mean_intensity)
    }


def compute_connected_3point(
    images: NDArray,
    config: CorrelatorConfig,
    sampling_config: SamplingConfig | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = True
) -> dict:
    """Compute 3-point connected correlator (cumulant).

    C₃(r₁, r₂) = ⟨I(x) I(x+r₁) I(x+r₂)⟩
                 - ⟨I(x)⟩ ⟨I(x+r₁) I(x+r₂)⟩
                 - ⟨I(x+r₁)⟩ ⟨I(x) I(x+r₂)⟩
                 - ⟨I(x+r₂)⟩ ⟨I(x) I(x+r₁)⟩
                 + 2⟨I(x)⟩ ⟨I(x+r₁)⟩ ⟨I(x+r₂)⟩

    For centered images (⟨I⟩ = 0), this simplifies to:
    C₃(r₁, r₂) = ⟨I(x) I(x+r₁) I(x+r₂)⟩

    Args:
        images: Image batch, shape (n_images, height, width)
        config: Correlator configuration
        sampling_config: Sampling configuration
        rng: Random number generator
        verbose: Whether to show progress bar

    Returns:
        Dictionary containing:
            - 'C3': Connected 3-point correlator values, shape (n_configs,)
            - 'r1_vectors': First separation vectors, shape (n_configs, 2)
            - 'r2_vectors': Second separation vectors, shape (n_configs, 2)
            - 'errors': Standard errors, shape (n_configs,)
            - 'mean_intensity': Mean image intensity
    """
    if rng is None:
        rng = np.random.default_rng()
    if sampling_config is None:
        sampling_config = SamplingConfig()

    # Normalize images
    images = images.astype(np.float32)
    mean_intensity = np.mean(images)
    images_centered = images - mean_intensity

    image_shape = images.shape[1:]

    # Generate separation vector pairs (at least n_configurations)
    separation_pairs = generate_separation_vectors(
        sampling_config, n_total=config.n_configurations, rng=rng
    )
    # Subsample if we generated more than requested
    if len(separation_pairs) > config.n_configurations:
        indices = rng.choice(len(separation_pairs), config.n_configurations, replace=False)
        separation_pairs = separation_pairs[indices]

    # Compute correlators
    C3_values = []
    errors = []
    r1_list = []
    r2_list = []

    iterator = enumerate(separation_pairs)
    if verbose:
        iterator = tqdm(list(iterator), desc="Computing C3")

    for idx, (r1, r2) in iterator:
        # Sample anchor points
        x_anchors = sample_anchor_points(
            image_shape, r1, r2, config.n_samples_per_config, rng
        )

        if len(x_anchors) == 0:
            C3_values.append(np.nan)
            errors.append(np.nan)
            r1_list.append(r1)
            r2_list.append(r2)
            continue

        # Compute triplet products
        products = compute_triplet_products(images_centered, x_anchors, r1, r2)

        # For centered images, C3 = ⟨I(x) I(x+r1) I(x+r2)⟩
        C3 = np.mean(products)
        C3_error = np.std(products) / np.sqrt(products.size)

        C3_values.append(C3)
        errors.append(C3_error)
        r1_list.append(r1)
        r2_list.append(r2)

    return {
        'C3': np.array(C3_values, dtype=np.float32),
        'r1_vectors': np.array(r1_list, dtype=np.float32),
        'r2_vectors': np.array(r2_list, dtype=np.float32),
        'errors': np.array(errors, dtype=np.float32),
        'mean_intensity': float(mean_intensity)
    }


def compute_per_digit_correlators(
    images: NDArray,
    labels: NDArray,
    config: CorrelatorConfig,
    sampling_config: SamplingConfig | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = True
) -> dict[int, dict]:
    """Compute N=2,3 correlators separately for each digit class.

    Args:
        images: Image batch, shape (n_images, height, width)
        labels: Digit labels, shape (n_images,)
        config: Correlator configuration
        sampling_config: Sampling configuration
        rng: Random number generator
        verbose: Whether to show progress

    Returns:
        Dictionary mapping digit (0-9) to correlator results:
            {digit: {'C2': ..., 'C3': ...}}
    """
    if rng is None:
        rng = np.random.default_rng()

    results = {}
    digits = np.unique(labels)

    for digit in digits:
        if verbose:
            print(f"\nProcessing digit {digit}...")

        # Filter images for this digit
        mask = labels == digit
        digit_images = images[mask]

        # Compute C2 and C3
        C2_result = compute_connected_2point(
            digit_images, config, sampling_config, rng, verbose=verbose
        )
        C3_result = compute_connected_3point(
            digit_images, config, sampling_config, rng, verbose=verbose
        )

        results[int(digit)] = {
            'C2': C2_result,
            'C3': C3_result
        }

    return results
