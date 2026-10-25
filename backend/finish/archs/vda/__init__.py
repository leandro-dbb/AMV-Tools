"""Video Depth Anything (Small) network, inference only.

Vendored from https://github.com/DepthAnything/Video-Depth-Anything
(Apache-2.0, (c) 2025 Bytedance Ltd. and/or its affiliates; LICENSE in this
folder): ``dinov2*``, ``dpt*``, ``motion_module``, ``util/blocks.py``.
Changes: EasyDict replaced by a dict (no extra dependency), the "xFormers
not available" prints removed; the model class keeps the upstream
``forward`` so ``video_depth_anything_vits.pth`` loads with strict=True.
Only the Small checkpoint is Apache-2.0 (Base/Large are CC-BY-NC-4.0).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal

INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
INTERP_LEN = 8


class VideoDepthAnything(nn.Module):
    def __init__(self, encoder="vits", features=64, out_channels=(48, 96, 192, 384),
                 use_bn=False, use_clstoken=False, num_frames=32, pe="ape"):
        super().__init__()
        self.intermediate_layer_idx = {"vits": [2, 5, 8, 11], "vitb": [2, 5, 8, 11], "vitl": [4, 11, 17, 23]}
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=list(out_channels),
                                    use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)

    def forward(self, x):
        """(B, T, 3, H, W) normalised, H/W multiples of 14 -> (B, T, H, W)
        relative disparity."""
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0, 1), self.intermediate_layer_idx[self.encoder],
                                                           return_class_token=True)
        depth = self.head(features, patch_h, patch_w, T)[0]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T))
