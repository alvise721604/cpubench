import time
import math
import numpy as np
from typing import Tuple

#___________________________________________________________________________________________________________________
def riemann_sinx_integral_vectorized(limit: float, step: float) -> Tuple[float, float]:
    start_time = time.time()
    x = np.arange(0.0, limit, step, dtype=np.float64)

    y = np.ones_like(x)
    mask = x != 0.0
    y[mask] = np.sin(x[mask]) / x[mask]

    result = y.sum() * step
    duration = time.time() - start_time
    return duration, result * 2.0

#___________________________________________________________________________________________________________________
def gaussian_integral_numpy_vectorized(limit: float, step: float) -> Tuple[float, float]:
    start_time = time.time()
    x = np.arange(0.0, limit, step, dtype=np.float64)
    result = np.exp(-(x * x)).sum() * step
    result *= 2.0
    duration = time.time() - start_time
    return duration, result * result