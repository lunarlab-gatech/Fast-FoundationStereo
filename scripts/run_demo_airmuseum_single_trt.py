# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Run Fast FoundationStereo (single ONNX model or TRT engine, from
make_single_onnx.py) over an AirMuseum rosbag sequence.

Copied from run_demo_single_trt.py; swaps static left/right PNG loading for
robotdataprocess bag loading, and loops over every synced stereo frame
instead of just one pair.

Supports two backends:
  - ONNX Runtime (default if --model_file points to an .onnx, or auto-detected)
  - TensorRT     (if --model_file points to an .engine)

The model expects ImageNet-normalised inputs, so this script applies
normalisation during preprocessing.

Usage:
  # Run directly with ONNX (no trtexec step needed):
  python run_demo_airmuseum_single_trt.py \
      --model_dir ./output_single_onnx \
      --dataset_path ~/data/AirMuseum_dataset/Scenario5 \
      --robot_name robotA

  # Or with an explicit model file:
  python run_demo_airmuseum_single_trt.py \
      --model_dir  ./output_single_onnx \
      --model_file ./output_single_onnx/fast_foundationstereo.onnx \
      --dataset_path ~/data/AirMuseum_dataset/Scenario5 \
      --robot_name robotA
"""

import argparse
import logging
import os
import sys
from decimal import Decimal
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import tqdm
import yaml

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from Utils import (
    set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
from robotdataprocess import CameraData, ImageData, ImageDataOnDisk

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Per-robot camera wiring
ROBOT_LEFT_CAM_MAP = {
    'drone': 'cam100',
    'robotA': 'cam101',
    'robotB': 'cam101',
    'robotC': 'cam101',
}
CAM_ID_TO_CALIB_NAME = {'cam100': 'cam0', 'cam101': 'cam1'}
CAM_ID_TO_BAG_NAME = {'cam100': 'cam100_imu.bag', 'cam101': 'cam101.bag'}


class SingleEngineTrtRunner:
    """Minimal TensorRT runner for a single engine with named I/O."""

    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f'Failed to deserialize TRT engine from {engine_path}. '
                f'This usually means the engine was built with a different '
                f'TensorRT version (yours: {trt.__version__}). '
                f'Rebuild with:  trtexec --onnx=<your .onnx> '
                f'--saveEngine={engine_path} --fp16')
        self.context = self.engine.create_execution_context()

    def _trt_to_torch_dtype(self, dt):
        trt = self.trt
        mapping = {
            trt.DataType.FLOAT:  torch.float32,
            trt.DataType.HALF:   torch.float16,
            trt.DataType.BF16:   torch.bfloat16,
            trt.DataType.INT32:  torch.int32,
            trt.DataType.INT8:   torch.int8,
            trt.DataType.BOOL:   torch.bool,
        }
        if dt not in mapping:
            raise RuntimeError(f'Unsupported TRT dtype: {dt}')
        return mapping[dt]

    def __call__(self, inputs: dict) -> dict:
        """Run inference.

        Args:
            inputs: {binding_name: torch.Tensor} for every input tensor.
        Returns:
            {binding_name: torch.Tensor} for every output tensor.
        """
        trt = self.trt

        for name, tensor in inputs.items():
            expected = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            if tensor.dtype != expected:
                inputs[name] = tensor.to(expected)
            if not inputs[name].is_contiguous():
                inputs[name] = inputs[name].contiguous()
            self.context.set_input_shape(name, tuple(inputs[name].shape))

        out_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
               == trt.TensorIOMode.OUTPUT
        ]

        outputs = {}
        for name in out_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, device='cuda', dtype=dtype)

        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        assert self.context.execute_async_v3(stream)

        return outputs


class OnnxRuntimeRunner:
    """Run inference via ONNX Runtime (GPU if available, else CPU)."""

    def __init__(self, onnx_path):
        import onnxruntime as ort
        providers = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        logging.info(f'ONNX Runtime providers: {providers}.')
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def __call__(self, inputs: dict) -> dict:
        feed = {}
        for name in self.input_names:
            tensor = inputs[name]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().float().numpy()
            feed[name] = tensor
        raw_outputs = self.session.run(self.output_names, feed)
        outputs = {}
        for name, arr in zip(self.output_names, raw_outputs):
            outputs[name] = torch.as_tensor(arr).cuda()
        return outputs


def normalize_imagenet(img_uint8: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization: (img/255 - mean) / std."""
    return ((img_uint8.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD


def resolve_config(model_path: str) -> str:
    """Find the YAML config matching the model file, falling back to defaults."""
    model_dir = os.path.dirname(model_path)
    base = os.path.splitext(os.path.basename(model_path))[0]
    candidates = [
        os.path.join(model_dir, f'{base}.yaml'),
        os.path.join(model_dir, 'config.yaml'),
        os.path.join(model_dir, 'onnx.yaml'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'No .yaml config found for {model_path}. '
        'Run make_single_onnx.py first.')


def find_model(model_dir: str) -> str:
    """Find an .engine or .onnx file in the directory (prefer .engine)."""
    for ext in ('.engine', '.onnx'):
        for f in os.listdir(model_dir):
            if f.endswith(ext):
                return os.path.join(model_dir, f)
    raise FileNotFoundError(
        f'No .engine or .onnx file found in {model_dir}. '
        'Run make_single_onnx.py first.')


def load_airmuseum_stereo(dataset_path: str, robot_name: str):
    """Load and rectify a synced left/right image sequence for one AirMuseum robot."""
    dataset_path = Path(dataset_path).expanduser()
    dataset_config_path = dataset_path.parent
    input_path = dataset_path / 'data' / robot_name

    left_cam_id = ROBOT_LEFT_CAM_MAP[robot_name]
    right_cam_id = 'cam101' if left_cam_id == 'cam100' else 'cam100'
    left_bag = input_path / CAM_ID_TO_BAG_NAME[left_cam_id]
    right_bag = input_path / CAM_ID_TO_BAG_NAME[right_cam_id]

    calib_name = f'{robot_name}_cameras_calib.yaml'
    cam_left, cam_right = CameraData.from_kalibr_stereo(
        dataset_config_path / 'sensors' / calib_name,
        CAM_ID_TO_CALIB_NAME[left_cam_id], CAM_ID_TO_CALIB_NAME[right_cam_id], alpha=0.0)

    left_data = ImageDataOnDisk.from_ros1_bag(left_bag, f'/{robot_name}/{left_cam_id}/image_raw')
    right_data = ImageDataOnDisk.from_ros1_bag(right_bag, f'/{robot_name}/{right_cam_id}/image_raw')
    assert left_data.encoding == right_data.encoding, 'Left/Right image encodings must match!'

    CameraData.align_ImageData_and_CameraData_to_imu_ts([left_data], cam_left)
    CameraData.align_ImageData_and_CameraData_to_imu_ts([right_data], cam_right)

    ImageDataOnDisk.crop_to_matched(left_data, right_data, Decimal('0.01'))
    ImageDataOnDisk.undistort_imagery_stereo(left_data, right_data, cam_left, cam_right)

    # Converts Mono8 -> RGB8 if needed; no-op if already RGB8.
    left_data.to_encoding(ImageData.ImageEncoding.RGB8)
    right_data.to_encoding(ImageData.ImageEncoding.RGB8)

    return left_data, right_data, cam_left, cam_right


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Fast FoundationStereo with ONNX Runtime or TensorRT')
    parser.add_argument('--model_dir', type=str,
                        default=f'output_320x480',
                        help='Directory containing .onnx/.engine + config.yaml')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Explicit path to .onnx or .engine file (overrides auto-search '
                             'in --model_dir, which prefers .engine over .onnx)')
    parser.add_argument('--dataset_path', type=str, default='~/data/AirMuseum_dataset/Scenario5',
                        help='e.g. ~/data/AirMuseum_dataset/Scenario5')
    parser.add_argument('--robot_name', type=str, default='robotC',
                        help='e.g. drone, robotA, robotB, robotC')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Defaults to <dataset_path>/results/Fast-FoundationStereo')
    parser.add_argument('--remove_invisible', type=int, default=1)
    parser.add_argument('--denoise_cloud', type=int, default=1)
    parser.add_argument('--denoise_nb_points', type=int, default=30)
    parser.add_argument('--denoise_radius', type=float, default=0.03)
    parser.add_argument('--save_vis', type=int, default=0,
                        help='Save disp_vis_*.png (left|right|colorized-disparity) per frame')
    parser.add_argument('--get_pc', type=int, default=0,
                        help='Generate and save point cloud')
    parser.add_argument('--zfar', type=float, default=100,
                        help='Max depth (m) to include in point cloud')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='Limit number of frames processed; -1 = all synced frames')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    if args.out_dir is None:
        args.out_dir = Path(args.dataset_path).expanduser() / 'results' / 'Fast-FoundationStereo'
    out_dir = os.path.join(args.out_dir, args.robot_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Find model and config ─────────────────────────────────────────────
    model_path = args.model_file if args.model_file else find_model(args.model_dir)
    cfg_path = resolve_config(model_path)
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    target_h, target_w = cfg['image_size']
    logging.info(f'Model target resolution: {target_h} x {target_w}.')

    # ── Load model (ONNX Runtime or TensorRT) ────────────────────────────
    logging.info(f'Loading model: {model_path}.')
    if model_path.endswith('.onnx'):
        runner = OnnxRuntimeRunner(model_path)
    else:
        runner = SingleEngineTrtRunner(model_path)

    # ── Load & rectify the AirMuseum stereo sequence ──────────────────────
    logging.info(f'Loading AirMuseum data for {args.robot_name} from {args.dataset_path}.')
    left_data, right_data, cam_left, cam_right = load_airmuseum_stereo(args.dataset_path, args.robot_name)
    n_frames = left_data.len()
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
    logging.info(f'{n_frames} synced stereo frames to process.')

    # Rectified intrinsics/baseline (undistort_imagery_stereo leaves each
    # camera's P describing the rectified projection, baseline included).
    K = cam_left.K.copy()
    baseline = -cam_right.P[0, 3] / cam_right.P[0, 0]
    logging.info(f'Rectified baseline: {baseline:.4f} m.')

    # NOTE on scale/K: `scale_x`/`scale_y` below are RESIZE ratios
    # (target/native), not focal lengths, despite run_demo_single_trt.py
    # naming the equivalent variables `fx`/`fy` (a naming collision with the
    # focal-length symbol that this script avoids). K_scaled is K rescaled to
    # match the target-resolution pixel grid, which depth2xyzmap() (Utils.py)
    # indexes directly -- so K must describe that grid for x/y to come out
    # right.
    orig_h, orig_w = cam_left.height, cam_left.width
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    if scale_x != 1 or scale_y != 1:
        logging.info(
            f'Resizing images: {orig_h}x{orig_w} -> {target_h}x{target_w} '
            f'(scale_x={scale_x:.4f}, scale_y={scale_y:.4f}).')
    K_scaled = K.copy()
    K_scaled[:2] *= np.array([scale_x, scale_y], dtype=np.float32)[:, np.newaxis]

    for i in tqdm.tqdm(range(n_frames), desc=f'{args.robot_name} inference'):
        # ── Read images (already RGB8 via load_airmuseum_stereo) ───────
        img0 = left_data.images[i]
        img1 = right_data.images[i]

        # ── Resize to model resolution (direct stretch) ────────────────
        if scale_x != 1 or scale_y != 1:
            img0 = cv2.resize(img0, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        H, W = img0.shape[:2]

        img0_ori = img0.copy()
        img1_ori = img1.copy()

        # ── Preprocess: ImageNet normalize → NCHW float tensor ────────
        img0_norm = normalize_imagenet(img0)
        img1_norm = normalize_imagenet(img1)

        t_left  = torch.as_tensor(img0_norm).cuda().float()[None].permute(0, 3, 1, 2)
        t_right = torch.as_tensor(img1_norm).cuda().float()[None].permute(0, 3, 1, 2)

        # ── Inference ──────────────────────────────────────────────
        outputs = runner({'left_image': t_left, 'right_image': t_right})
        disp = outputs['disparity']

        # NOTE: disp stays on the target-resolution grid, in target-resolution
        # pixel units -- this is what run_demo_single_trt.py does NOT do:
        # that script rescales disp by (1/fx) back to native-pixel units
        # while still pairing it with the target-scaled K (K_scaled here),
        # which double-corrects for the resize and reports depth off by a
        # factor of scale_x. Since K_scaled is already the target-resolution
        # K, disp must stay in target-resolution units too, for
        # `depth = K_scaled[0,0] * baseline / disp` to be dimensionally
        # consistent.
        disp = disp.float().cpu().numpy().reshape(H, W).clip(0, None)

        # Visualise disparity (non-blocking; updates live per frame)
        vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
        vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
        if args.save_vis:
            imageio.imwrite(f'{out_dir}/disp_vis_{i:06d}.png', vis)
        s = 1280 / vis.shape[1]
        resized_vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)))
        cv2.imshow('disp', resized_vis[:, :, ::-1])
        cv2.waitKey(1)

        # Remove invisible pixels
        if args.remove_invisible:
            _, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            invalid = (xx - disp) < 0
            disp[invalid] = np.inf

        # Depth (meters), saved every frame regardless of --get_pc 
        depth = (K_scaled[0, 0] * baseline / disp).astype(np.float32)
        depth_dir = os.path.join(out_dir, 'depth')
        os.makedirs(depth_dir, exist_ok=True)
        np.save(f'{depth_dir}/{left_data.timestamps[i]}.npy', depth)

        # Point cloud generation
        if args.get_pc:
            xyz_map = depth2xyzmap(depth, K_scaled)
            pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
            pts = np.asarray(pcd.points)
            keep = (pts[:, 2] > 0) & (pts[:, 2] <= args.zfar)
            pcd = pcd.select_by_index(np.where(keep)[0])
            o3d.io.write_point_cloud(f'{out_dir}/cloud_{i:06d}.ply', pcd)

            if args.denoise_cloud:
                _, ind = pcd.remove_radius_outlier(
                    nb_points=args.denoise_nb_points,
                    radius=args.denoise_radius)
                pcd = pcd.select_by_index(ind)
                o3d.io.write_point_cloud(f'{out_dir}/cloud_denoise_{i:06d}.ply', pcd)

    logging.info('Done.')