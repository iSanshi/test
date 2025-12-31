"""
Gaussian Process utilities for preference learning.
"""

from .gaussian_process import GaussianProcess
from .audio_gp import AudioPreferenceGaussianProcess

__all__ = ["GaussianProcess", "AudioPreferenceGaussianProcess"]
