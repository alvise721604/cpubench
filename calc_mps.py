import time
from typing import Tuple
import torch

#___________________________________________________________________________________________________________________
def riemann_sinx_integral_vectorized(limit: float, step: float) -> Tuple[float, float]:
    start_time = time.time()
    
    # Create a tensor of values from 0 to limit with the given step size
    x = torch.arange(0.0, limit, step, dtype=torch.float64)

    # Initialize y as ones and apply mask for non-zero elements
    y = torch.ones_like(x)
    mask = x != 0.0
    y[mask] = torch.sin(x[mask]) / x[mask]

    result = y.sum() * step
    duration = time.time() - start_time
    return duration, result * 2.0

#___________________________________________________________________________________________________________________
def gaussian_integral_pytorch_vectorized(limit: float, step: float) -> Tuple[float, float]:
    start_time = time.time()
    
    # Create a tensor of values from 0 to limit with the given step size
    x = torch.arange(0.0, limit, step, dtype=torch.float64)
    
    # Compute the Gaussian integral using PyTorch operations
    result = torch.exp(-(x * x)).sum() * step
    result *= 2.0
    duration = time.time() - start_time
    
    return duration, result * result

# Example usage:
if __name__ == "__main__":
    limit = 10.0
    step = 0.01
    print(riemann_sinx_integral_vectorized(limit, step))
    print(gaussian_integral_pytorch_vectorized(limit, step))
