import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import argparse
import torch
import mmcv
import copy
import numpy as np
from mmcv import Config

# Use our own plugin loader instead of mmdeploy
try:
    from mmdeploy.backend.tensorrt import load_tensorrt_plugin
except ImportError:
    from det2trt.utils.plugin_loader import load_tensorrt_plugin

import sys

sys.path.append(".")
from det2trt.utils.tensorrt import (
    get_logger,
    create_engine_context,
    allocate_buffers,
    do_inference,
)
from third_party.bev_mmdet3d.models.builder import build_model
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("trt_model", help="checkpoint file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    load_tensorrt_plugin()

    trt_model = args.trt_model
    config_file = args.config
    TRT_LOGGER = get_logger(trt.Logger.INTERNAL_ERROR)

    engine, context = create_engine_context(trt_model, TRT_LOGGER)

    stream = cuda.Stream()

    config = Config.fromfile(config_file)
    if hasattr(config, "plugin"):
        import importlib
        import sys

        sys.path.append(".")
        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    output_shapes = config.output_shapes
    input_shapes = config.input_shapes
    default_shapes = config.default_shapes

    for key in default_shapes:
        if key in locals():
            raise RuntimeError(f"Variable {key} has been defined.")
        locals()[key] = default_shapes[key]

    dataset = build_dataset(cfg=config.data.val)
    
    # Get mini_val sample tokens for filtering
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
    mini_val_scenes = splits.mini_val
    mini_val_sample_tokens = set()
    for scene in nusc.scene:
        if scene['name'] in mini_val_scenes:
            sample_token = scene['first_sample_token']
            while sample_token:
                mini_val_sample_tokens.add(sample_token)
                sample = nusc.get('sample', sample_token)
                sample_token = sample['next']
    
    # Filter dataset to only mini_val samples
    original_data_infos = dataset.data_infos
    filtered_data_infos = [info for info in original_data_infos if info['token'] in mini_val_sample_tokens]
    dataset.data_infos = filtered_data_infos
    print(f'Filtering dataset to {len(dataset.data_infos)} mini_val samples (from {len(original_data_infos)} total)')
    
    # Rebuild dataloader with filtered dataset
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )

    pth_model = build_model(config.model, test_cfg=config.get("test_cfg"))

    ts = []
    bbox_results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    prev_bev = np.random.randn(config.bev_h_ * config.bev_w_, 1, config._dim_)
    prev_frame_info = {
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }
    for data in loader:
        img = data["img"][0].data[0].numpy()
        img_metas = data["img_metas"][0].data[0]

        use_prev_bev = np.array([1.0])
        if img_metas[0]["scene_token"] != prev_frame_info["scene_token"]:
            use_prev_bev = np.array([0.0])
        prev_frame_info["scene_token"] = img_metas[0]["scene_token"]
        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        if use_prev_bev[0] == 1:
            img_metas[0]["can_bus"][:3] -= prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= prev_frame_info["prev_angle"]
        else:
            img_metas[0]["can_bus"][-1] = 0
            img_metas[0]["can_bus"][:3] = 0
        can_bus = img_metas[0]["can_bus"]
        lidar2img = np.stack(img_metas[0]["lidar2img"], axis=0)
        batch_size, cameras, _, img_h, img_w = img.shape

        output_shapes_ = {}
        for key in output_shapes.keys():
            shape = output_shapes[key][:]
            for shape_i in range(len(shape)):
                if isinstance(shape[shape_i], str):
                    shape[shape_i] = eval(shape[shape_i])
            output_shapes_[key] = shape

        input_shapes_ = {}
        for key in input_shapes.keys():
            shape = input_shapes[key][:]
            for shape_i in range(len(shape)):
                if isinstance(shape[shape_i], str):
                    shape[shape_i] = eval(shape[shape_i])
            input_shapes_[key] = shape

        inputs, outputs, bindings = allocate_buffers(
            engine, context, input_shapes=input_shapes_, output_shapes=output_shapes_
        )

        for inp in inputs:
            if inp.name == "image":
                inp.host = img.reshape(-1).astype(np.float32)
            elif inp.name == "prev_bev":
                inp.host = prev_bev.reshape(-1).astype(np.float32)
            elif inp.name == "use_prev_bev":
                inp.host = use_prev_bev.reshape(-1).astype(np.float32)
            elif inp.name == "can_bus":
                inp.host = can_bus.reshape(-1).astype(np.float32)
            elif inp.name == "lidar2img":
                inp.host = lidar2img.reshape(-1).astype(np.float32)
            else:
                raise RuntimeError(f"Cannot find input name {inp.name}.")
        
        # Debug: check input values for first sample
        if len(bbox_results) == 0:
            print("\n=== Checking inputs ===")
            for inp in inputs:
                data = inp.host[:10] if len(inp.host) > 10 else inp.host
                print(f"Input {inp.name}: shape={inp.host.shape}, has_nan={np.isnan(inp.host).any()}, "
                      f"min={np.nanmin(inp.host):.4f}, max={np.nanmax(inp.host):.4f}")
            print("="*50)

        trt_outputs, t = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        trt_outputs = {
            out.name: out.host.reshape(*output_shapes_[out.name]) for out in trt_outputs
        }

        prev_bev = trt_outputs.pop("bev_embed")
        prev_frame_info["prev_pos"] = tmp_pos
        prev_frame_info["prev_angle"] = tmp_angle

        trt_outputs = {k: torch.from_numpy(v) for k, v in trt_outputs.items()}
        
        # Debug: check TRT output values
        if len(bbox_results) == 0:
            for k, v in trt_outputs.items():
                print(f"TRT output {k}: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

        result = pth_model.post_process(**trt_outputs, img_metas=img_metas)
        
        # Debug: check post-process output
        if len(bbox_results) == 0:
            print(f"Post-process result: {len(result[0]['pts_bbox']['boxes_3d'])} boxes")
            if len(result[0]['pts_bbox']['boxes_3d']) > 0:
                print(f"  Scores: min={result[0]['pts_bbox']['scores_3d'].min():.4f}, max={result[0]['pts_bbox']['scores_3d'].max():.4f}")
        
        bbox_results.extend(result)
        ts.append(t)

        for _ in range(len(img)):
            prog_bar.update()
    
    print(f'\nTotal results: {len(bbox_results)}')
    print(f'Non-None results: {sum(1 for r in bbox_results if r is not None)}')
    
    # Check how many results have detections
    num_with_boxes = sum(1 for r in bbox_results if len(r['pts_bbox']['boxes_3d']) > 0)
    total_boxes = sum(len(r['pts_bbox']['boxes_3d']) for r in bbox_results)
    print(f'Results with boxes: {num_with_boxes}/{len(bbox_results)}')
    print(f'Total boxes across all results: {total_boxes}')
    
    if bbox_results:
        print(f'First result type: {type(bbox_results[0])}')
        print(f'First result keys: {bbox_results[0].keys() if isinstance(bbox_results[0], dict) else "NOT A DICT"}')
        # Check first result's boxes
        first_boxes = bbox_results[0]['pts_bbox']['boxes_3d']
        print(f'First result boxes type: {type(first_boxes)}')
        print(f'First result num boxes: {len(first_boxes)}')
        print(f'First result scores shape: {bbox_results[0]["pts_bbox"]["scores_3d"].shape}')
        print(f'First result labels shape: {bbox_results[0]["pts_bbox"]["labels_3d"].shape}')

    metric = dataset.evaluate(bbox_results)

    # summary
    print("*" * 50 + " SUMMARY " + "*" * 50)
    for key in metric.keys():
        if key == "pts_bbox_NuScenes/NDS":
            print(f"NDS: {round(metric[key], 3)}")
        elif key == "pts_bbox_NuScenes/mAP":
            print(f"mAP: {round(metric[key], 3)}")

    latency = round(sum(ts[1:-1]) / len(ts[1:-1]) * 1000, 2)
    print(f"Latency: {latency}ms")
    print(f"FPS: {1000 / latency}")


if __name__ == "__main__":
    main()
