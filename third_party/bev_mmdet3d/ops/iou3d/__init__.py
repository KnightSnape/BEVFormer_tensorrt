# Conditional import to avoid missing CUDA extension during ONNX conversion
try:
    from .iou3d_utils import boxes_iou_bev, nms_gpu, nms_normal_gpu
    __all__ = ["boxes_iou_bev", "nms_gpu", "nms_normal_gpu"]
except ImportError as e:
    print(f"Warning: iou3d CUDA extension not available: {e}")
    __all__ = []
