# Compatibility layer for mmcv.runner decorators
"""
Provides compatibility for mmcv.runner decorators that have been removed in newer versions.
"""

try:
    from mmcv.runner import force_fp32, auto_fp16, BaseModule, get_dist_info
except ImportError:
    # Fallback implementations for newer mmengine/mmcv versions
    try:
        from mmengine.model import BaseModule
    except ImportError:
        from torch.nn import Module as BaseModule
    
    try:
        from mmengine.dist import get_dist_info
    except ImportError:
        def get_dist_info():
            return 0, 1
    
    # Simple pass-through decorators for fp16/fp32
    # In practice, we don't need these for ONNX export
    def force_fp32(apply_to=None, out_fp16=False):
        """Decorator to force FP32 precision (no-op in compatibility mode)."""
        def decorator(func):
            return func
        return decorator
    
    def auto_fp16(apply_to=None, out_fp32=False):
        """Decorator to enable auto FP16 (no-op in compatibility mode)."""
        def decorator(func):
            return func
        return decorator

# DataContainer compatibility
try:
    from mmcv.parallel.data_container import DataContainer
except ImportError:
    # Simple wrapper for data
    class DataContainer:
        def __init__(self, data, **kwargs):
            self._data = data
        
        @property
        def data(self):
            return self._data

__all__ = ['force_fp32', 'auto_fp16', 'BaseModule', 'get_dist_info', 'DataContainer']
