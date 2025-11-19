"""
Utility math functions for ScratchFormer.

Contains:
- stable_softmax: numerically stable softmax
- gelu: Gaussian Error Linear Unit activation
- make_padding_mask: mask out padding tokens
- make_causal_mask: lower-triangular mask to prevent attention to future tokens
- combine_masks: combine padding + causal masks into a single boolean mask

All masks are boolean where True indicates allowed positions (i.e., keep),
and False indicates masked positions (i.e., block).
"""

from typing import Optional;
import numpy as np;

def to_numpy(x):
    """If x is already a NumPy array, it simply returns it unchanged."""
    if isinstance(x, np.ndarray):
        return x
    """If x is not a NumPy array, it converts it into one using np.array(x)."""
    return np.array(x)


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert input to numpy array, 64-bit"""
    x = to_numpy(x).astype(np.float64)
    """Find the max value along a given axis"""
    x_max = np.max(x, axis=axis, keepdims=True)
    """Exponentiate after subtracting max"""
    exps = np.exp(x - x_max)
    """Sum all exponent values along the same axis"""
    sums = np.sum(exps, axis=axis, keepdims=True)
    """Compute the softmax"""
    return (exps / sums).astype(np.float32)
