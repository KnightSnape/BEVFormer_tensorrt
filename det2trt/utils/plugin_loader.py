"""
Helper utility to load custom TensorRT plugins.
Replaces mmdeploy.backend.tensorrt.load_tensorrt_plugin for standalone usage.
"""

import os
import ctypes


def load_tensorrt_plugin(plugin_path=None):
    """
    Load TensorRT plugin library.
    
    Args:
        plugin_path (str, optional): Path to the plugin .so file. 
                                     If None, looks for libtensorrt_ops.so in default locations.
    """
    if plugin_path is None:
        # Try to find the plugin in common locations (prioritize TensorRT/lib over build)
        possible_paths = [
            "TensorRT/lib/libtensorrt_ops.so",
            "./TensorRT/lib/libtensorrt_ops.so",
            "/workspace/TensorRT/lib/libtensorrt_ops.so",
            "TensorRT/build/libtensorrt_ops.so",
            "./TensorRT/build/libtensorrt_ops.so",
            "/workspace/TensorRT/build/libtensorrt_ops.so",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                plugin_path = path
                break
    
    if plugin_path is None:
        print("Warning: Failed to load TensorRT plugins: cannot find libtensorrt_ops.so")
        return
    
    if not os.path.exists(plugin_path):
        print(f"Warning: Plugin library not found at: {plugin_path}")
        return
    
    # Load the plugin library
    try:
        ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
        print(f"Loaded tensorrt plugins from {plugin_path}")
    except Exception as e:
        print(f"Warning: Failed to load TensorRT plugins from {plugin_path}: {e}")
