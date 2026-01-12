import torch
import onnxruntime as ort
import numpy as np
import sys
sys.path.append(".")

try:
    from mmengine import Config
except ImportError:
    from mmcv import Config
from mmdet.models import build_detector
try:
    from mmengine.runner import load_checkpoint
except ImportError:
    from mmcv.runner import load_checkpoint


def test_onnx_vs_pytorch():
    config = Config.fromfile("configs/bevformer/bevformer_tiny_trt.py")
    
    # Import plugins
    if hasattr(config, 'plugin'):
        import importlib
        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)
    
    # Build PyTorch model
    model = build_detector(config.model, test_cfg=config.get("test_cfg", None))
    checkpoint = load_checkpoint(model, "checkpoints/pytorch/bevformer_tiny_epoch_24.pth", map_location="cpu")
    model.cuda().eval()
    model.forward = model.forward_trt
    
    # Prepare inputs
    batch = 1
    num_cams = 6
    img_h, img_w = 480, 800
    bev_h, bev_w = 50, 50
    embed_dims = 256
    
    torch.manual_seed(42)
    image = torch.randn(batch, num_cams, 3, img_h, img_w).cuda()
    prev_bev = torch.randn(bev_h * bev_w, batch, embed_dims).cuda()
    use_prev_bev = torch.tensor([0.0]).cuda()
    can_bus = torch.randn(18).cuda()
    lidar2img = torch.randn(batch, num_cams, 4, 4).cuda()
    
    # Test PyTorch
    with torch.no_grad():
        pth_bev, pth_classes, pth_coords = model(image, prev_bev, use_prev_bev, can_bus, lidar2img)
    
    print("=== PyTorch outputs ===")
    print(f"outputs_classes: shape={pth_classes.shape}, min={pth_classes.min():.4f}, max={pth_classes.max():.4f}, mean={pth_classes.mean():.4f}")
    print(f"outputs_coords: shape={pth_coords.shape}, min={pth_coords.min():.4f}, max={pth_coords.max():.4f}, mean={pth_coords.mean():.4f}")
    
    # Test ONNX
    onnx_model_path = "checkpoints/onnx/bevformer_tiny_epoch_24.onnx"
    sess = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    onnx_inputs = {
        'image': image.cpu().numpy(),
        'prev_bev': prev_bev.cpu().numpy(),
        'use_prev_bev': use_prev_bev.cpu().numpy(),
        'can_bus': can_bus.cpu().numpy(),
        'lidar2img': lidar2img.cpu().numpy()
    }
    
    onnx_outputs = sess.run(None, onnx_inputs)
    onnx_bev, onnx_classes, onnx_coords = onnx_outputs
    
    print("\n=== ONNX outputs ===")
    print(f"outputs_classes: shape={onnx_classes.shape}, min={onnx_classes.min():.4f}, max={onnx_classes.max():.4f}, mean={onnx_classes.mean():.4f}")
    print(f"outputs_coords: shape={onnx_coords.shape}, min={onnx_coords.min():.4f}, max={onnx_coords.max():.4f}, mean={onnx_coords.mean():.4f}")
    
    # Compare
    print("\n=== Comparison ===")
    print(f"classes diff: mean={np.abs(pth_classes.cpu().numpy() - onnx_classes).mean():.6f}, max={np.abs(pth_classes.cpu().numpy() - onnx_classes).max():.6f}")
    print(f"coords diff: mean={np.abs(pth_coords.cpu().numpy() - onnx_coords).mean():.6f}, max={np.abs(pth_coords.cpu().numpy() - onnx_coords).max():.6f}")


if __name__ == "__main__":
    test_onnx_vs_pytorch()
