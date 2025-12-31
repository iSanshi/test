"""
Utility math functions for the Gaussian Process implementation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import norm


def binary_entropy(probability: float) -> float:
    """Return the binary entropy for the given Bernoulli probability."""
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return -probability * np.log2(probability) - (1 - probability) * np.log2(1 - probability)


def normal_cdf(value: float, sigma: float = 1, *, scale: Optional[float] = None) -> float:
    """Cumulative distribution function of a normal distribution."""
    if scale is not None:
        sigma = scale
    return norm.cdf(value, scale=sigma)


def normal_pdf(value: float, sigma: float = 1, *, scale: Optional[float] = None) -> float:
    """Probability density function of a normal distribution."""
    if scale is not None:
        sigma = scale
    return norm.pdf(value, scale=sigma)


def normal_pdf_second_derivative(value: float, sigma: float = 1, *, scale: Optional[float] = None) -> float:
    """Second derivative of the normal PDF."""
    if scale is not None:
        sigma = scale
    scale_squared = sigma ** 2
    exponent = np.exp(-value ** 2 / (2 * scale_squared))
    return -value / scale_squared * exponent / (np.sqrt(2 * np.pi) * sigma)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

def h(probability: float) -> float:
    """Alias for :func:`binary_entropy`."""
    return binary_entropy(probability)


def phi(value: float, sigma: float = 1) -> float:
    """Alias for :func:`normal_cdf`."""
    return normal_cdf(value, sigma=sigma)


def phip(value: float, sigma: float = 1) -> float:
    """Alias for :func:`normal_pdf`."""
    return normal_pdf(value, sigma=sigma)


def phipp(value: float, sigma: float = 1) -> float:
    """Alias for :func:`normal_pdf_second_derivative`."""
    return normal_pdf_second_derivative(value, sigma=sigma)
