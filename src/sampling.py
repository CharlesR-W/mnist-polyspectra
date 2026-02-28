"""Spatial configuration sampling for N-point correlators."""

import numpy as np
from numpy.typing import NDArray

from .config import SamplingConfig


def generate_separation_vectors(
    config: SamplingConfig,
    rng: np.random.Generator | None = None
) -> NDArray:
    """Generate (r1, r2) separation vector pairs for 3-point correlators.

    Samples separation vectors with oversampling of special geometric configurations
    (equilateral, squeezed, collinear).

    Args:
        config: Sampling configuration
        rng: Random number generator (default: new Generator)

    Returns:
        Array of shape (n_configs, 2, 2) containing pairs of 2D separation vectors
    """
    if rng is None:
        rng = np.random.default_rng()

    vectors = []

    # Standard uniform sampling
    n_standard = int(config.n_angles * len(config.separation_grid))

    for r1_mag in config.separation_grid:
        for angle_idx in range(config.n_angles):
            angle1 = 2 * np.pi * angle_idx / config.n_angles
            r1 = np.array([r1_mag * np.cos(angle1), r1_mag * np.sin(angle1)])

            # Sample r2 uniformly
            r2_mag = rng.choice(config.separation_grid)
            angle2 = rng.uniform(0, 2 * np.pi)
            r2 = np.array([r2_mag * np.cos(angle2), r2_mag * np.sin(angle2)])

            vectors.append(np.stack([r1, r2]))

    # Oversample special geometries
    n_special = int(n_standard * (config.geometry_oversample - 1))

    for _ in range(n_special):
        geom = rng.choice(config.emphasize_geometries)
        r_mag = rng.choice(config.separation_grid)

        if geom == 'equilateral':
            # Equilateral triangle: |r1| = |r2| = |r2 - r1|
            angle1 = rng.uniform(0, 2 * np.pi)
            r1 = np.array([r_mag * np.cos(angle1), r_mag * np.sin(angle1)])
            r2 = np.array([
                r_mag * np.cos(angle1 + 2*np.pi/3),
                r_mag * np.sin(angle1 + 2*np.pi/3)
            ])
        elif geom == 'squeezed':
            # Squeezed: r1 ≈ r2 (small angle between them)
            angle1 = rng.uniform(0, 2 * np.pi)
            delta_angle = rng.uniform(-np.pi/8, np.pi/8)  # Small angle variation
            r1 = np.array([r_mag * np.cos(angle1), r_mag * np.sin(angle1)])
            r2 = np.array([
                r_mag * np.cos(angle1 + delta_angle),
                r_mag * np.sin(angle1 + delta_angle)
            ])
        elif geom == 'collinear':
            # Collinear: r1 and r2 parallel or antiparallel
            angle = rng.uniform(0, 2 * np.pi)
            r1_mag_local = rng.choice(config.separation_grid)
            r2_mag_local = rng.choice(config.separation_grid)

            if rng.random() < 0.5:  # Parallel
                r1 = np.array([r1_mag_local * np.cos(angle), r1_mag_local * np.sin(angle)])
                r2 = np.array([r2_mag_local * np.cos(angle), r2_mag_local * np.sin(angle)])
            else:  # Antiparallel
                r1 = np.array([r1_mag_local * np.cos(angle), r1_mag_local * np.sin(angle)])
                r2 = np.array([
                    r2_mag_local * np.cos(angle + np.pi),
                    r2_mag_local * np.sin(angle + np.pi)
                ])

        vectors.append(np.stack([r1, r2]))

    return np.array(vectors)


def sample_anchor_points(
    image_shape: tuple[int, int],
    r1: NDArray,
    r2: NDArray,
    n_samples: int,
    rng: np.random.Generator | None = None
) -> NDArray:
    """Sample valid anchor points for 3-point correlator computation.

    Ensures that x, x+r1, x+r2 all lie within image boundaries.

    Args:
        image_shape: Shape of image (height, width)
        r1: First separation vector, shape (2,)
        r2: Second separation vector, shape (2,)
        n_samples: Number of anchor points to sample
        rng: Random number generator

    Returns:
        Array of shape (n_samples, 2) containing valid anchor points
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image_shape
    r1 = np.round(r1).astype(int)
    r2 = np.round(r2).astype(int)

    # Compute valid range for anchor points
    min_y = max(0, -min(0, r1[1], r2[1]))
    max_y = min(h - 1, h - 1 - max(0, r1[1], r2[1]))
    min_x = max(0, -min(0, r1[0], r2[0]))
    max_x = min(w - 1, w - 1 - max(0, r1[0], r2[0]))

    if max_y < min_y or max_x < min_x:
        # No valid anchor points for this configuration
        return np.zeros((0, 2), dtype=int)

    # Sample uniformly from valid region
    y_samples = rng.integers(min_y, max_y + 1, size=n_samples)
    x_samples = rng.integers(min_x, max_x + 1, size=n_samples)

    return np.stack([x_samples, y_samples], axis=1)


def compute_triplet_products(
    images: NDArray,
    x_anchors: NDArray,
    r1: NDArray,
    r2: NDArray
) -> NDArray:
    """Compute I(x) * I(x+r1) * I(x+r2) for batch of images.

    Args:
        images: Image batch, shape (n_images, height, width)
        x_anchors: Anchor points, shape (n_samples, 2)
        r1: First separation vector, shape (2,)
        r2: Second separation vector, shape (2,)

    Returns:
        Array of shape (n_images, n_samples) containing triplet products
    """
    r1 = np.round(r1).astype(int)
    r2 = np.round(r2).astype(int)

    n_images = images.shape[0]
    n_samples = x_anchors.shape[0]

    if n_samples == 0:
        return np.zeros((n_images, 0), dtype=np.float32)

    products = np.zeros((n_images, n_samples), dtype=np.float32)

    for i, (x, y) in enumerate(x_anchors):
        I_x = images[:, y, x]
        I_x_r1 = images[:, y + r1[1], x + r1[0]]
        I_x_r2 = images[:, y + r2[1], x + r2[0]]
        products[:, i] = I_x * I_x_r1 * I_x_r2

    return products


def compute_pair_products(
    images: NDArray,
    x_anchors: NDArray,
    r: NDArray
) -> NDArray:
    """Compute I(x) * I(x+r) for batch of images (2-point correlator).

    Args:
        images: Image batch, shape (n_images, height, width)
        x_anchors: Anchor points, shape (n_samples, 2)
        r: Separation vector, shape (2,)

    Returns:
        Array of shape (n_images, n_samples) containing pair products
    """
    r = np.round(r).astype(int)

    n_images = images.shape[0]
    n_samples = x_anchors.shape[0]

    if n_samples == 0:
        return np.zeros((n_images, 0), dtype=np.float32)

    products = np.zeros((n_images, n_samples), dtype=np.float32)

    for i, (x, y) in enumerate(x_anchors):
        I_x = images[:, y, x]
        I_x_r = images[:, y + r[1], x + r[0]]
        products[:, i] = I_x * I_x_r

    return products
