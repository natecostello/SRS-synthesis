"""SRS Synthesis Package

Python implementations of SRS synthesis algorithms for generating acceleration 
time histories that match target shock response spectra.
"""

from .damped_sine import DSSynthesizer
from .wavelet import WSynthesizer

__version__ = "1.0.0"
__author__ = "Nate Costello"
__all__ = ["DSSynthesizer", "WSynthesizer"]
