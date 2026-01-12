try:
    from pytorch_quantization import nn as quant_nn
    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False
    print("Warning: pytorch_quantization not found. INT8 quantization will not be available.")

import os
import argparse
try:
    from mmengine import Config
except ImportError:
    from mmcv import Config

import sys

sys.path.append(".")
from det2trt.convert import pytorch2onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch to ONNX")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--opset_version", type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--flag", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint

    config = Config.fromfile(config_file)
    if hasattr(config, "plugin"):
        import importlib

        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    output = os.path.split(args.checkpoint)[1].split(".")[0]

    if args.int8:
        if not HAS_QUANT:
            raise ImportError(
                "pytorch_quantization is required for INT8 quantization. "
                "Please install it with: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
            )
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
    if args.flag:
        output += f"_{args.flag}"
    output_file = os.path.join(config.ONNX_PATH, output + ".onnx")

    pytorch2onnx(
        config,
        checkpoint=checkpoint_file,
        output_file=output_file,
        verbose=False,
        opset_version=args.opset_version,
        cuda=args.cuda,
    )


if __name__ == "__main__":
    main()
