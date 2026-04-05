import time
from typing import Tuple
import torch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ___________________________________________________________________________________________________________________
def riemann_sinx_integral_torch(limit: float, step: float) -> Tuple[float, float]:
    device = get_device()
    print(f"Called torch calculator on device: {device}")
    

    # MPS: meglio float32, non float64
    x = torch.arange(0.0, limit, step, dtype=torch.float32, device=device)
    torch.mps.synchronize()

    y = torch.ones_like(x)
    mask = x != 0.0
    torch.mps.synchronize()

    start_time = time.time()
    y[mask] = torch.sin(x[mask]) / x[mask]
    result = y.sum() * step

    # sincronizzazione per timing corretto su GPU
    if device.type == "mps":
        torch.mps.synchronize()

    duration = time.time() - start_time
    return duration, (result * 2.0).item()


# ___________________________________________________________________________________________________________________
def gaussian_integral_torch(limit: float, step: float) -> Tuple[float, float]:
    device = get_device()
    print(f"Called torch calculator on device: {device}")
    

    x = torch.arange(0.0, limit, step, dtype=torch.float32, device=device)
    torch.mps.synchronize()

    start_time = time.time()
    result = torch.exp(-(x * x)).sum() * step
    result *= 2.0
    result = result * result

    if device.type == "mps":
        torch.mps.synchronize()

    duration = time.time() - start_time
    return duration, result.item()


if __name__ == "__main__":
    limit = 10.0
    step = 0.01
    print(riemann_sinx_integral_torch(limit, step))
    print(gaussian_integral_torch(limit, step))