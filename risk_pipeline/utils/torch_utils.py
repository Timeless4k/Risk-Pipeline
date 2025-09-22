"""
PyTorch utilities for RiskPipeline with GPU/CPU helpers.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


def get_torch_device(prefer_gpu: bool = True) -> str:
    """Return device string for PyTorch operations."""
    if not TORCH_AVAILABLE:
        return 'cpu'
    if prefer_gpu and torch.cuda.is_available():
        try:
            _ = torch.cuda.get_device_name(0)
            return 'cuda:0'
        except Exception as err:
            logger.warning(f"CUDA present but unavailable: {err}; using CPU")
            return 'cpu'
    return 'cpu'


def torch_cuda_summary() -> Tuple[bool, str]:
    """Quick CUDA availability summary."""
    if not TORCH_AVAILABLE:
        return False, "PyTorch not installed"
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            return True, f"CUDA available: {name} (capability {cap[0]}.{cap[1]})"
        except Exception as err:
            return False, f"CUDA check error: {err}"
    return False, "CUDA not available"


def empty_cuda_cache() -> None:
    """Clear PyTorch CUDA cache if available."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


