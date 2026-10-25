"""RIFE 4.25 flow network (inference only).

Vendored from:
- Practical-RIFE, MIT License, (c) 2021 hzwer — https://github.com/hzwer/Practical-RIFE
- vs-rife ``IFNet_HDv3_v4_25.py`` / ``warplayer.py``, MIT License,
  (c) 2021 HolyWu — https://github.com/HolyWu/vs-rife

Changes vs. upstream: the feature ``Head`` runs inside ``IFNet.forward`` and
the backwarp grid is built lazily per input size, so callers only pass
``(img0, img1, timestep)``. Layer names are unchanged so the official
``flownet_v4.25.pkl`` loads with ``strict=True``.

Why 4.25: upstream recommends it "by default for most scenes" and its release
note calls out significantly improved anime scenes; at ~25 MB it is also the
lightest of the 4.2x line that keeps full quality.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Input sizes must be multiples of this (per-version constant from vs-rife).
MODULO = 64

# Practical-RIFE versions sharing this graph (vs-rife's IFNet_HDv3_v4_25*,
# IFNet_HDv3_v4_26*): Head output channels, block widths, coarse-to-fine
# scale list, input modulo. 4.26 is the 4.25 graph with new weights.
VARIANTS = {
    "4.25": {"enc": 4, "c4": 32, "scales": (16, 8, 4, 2, 1), "modulo": 64},
    "4.26": {"enc": 4, "c4": 32, "scales": (16, 8, 4, 2, 1), "modulo": 64},
    "4.25.lite": {"enc": 4, "c4": 24, "scales": (32, 16, 8, 4, 1), "modulo": 128},
    "4.26.heavy": {"enc": 16, "c4": 32, "scales": (16, 8, 4, 2, 1), "modulo": 64},
}


def _conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True),
    )


class Head(nn.Module):
    def __init__(self, out_ch: int = 4):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, out_ch, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = x.clamp(0.0, 1.0)
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(_conv(in_planes, c // 2, 3, 2, 1), _conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1.0):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear")
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


def _warp(ten_input, ten_flow, flow_div, grid):
    dtype = ten_input.dtype
    ten_flow = ten_flow.float()
    ten_flow = torch.cat([ten_flow[:, 0:1] / flow_div[0], ten_flow[:, 1:2] / flow_div[1]], 1)
    g = (grid + ten_flow).permute(0, 2, 3, 1)
    return F.grid_sample(ten_input.float(), g, mode="bilinear", padding_mode="border",
                         align_corners=True).to(dtype)


class IFNet(nn.Module):
    def __init__(self, scale: float = 1.0, variant: str = "4.25"):
        super().__init__()
        v = VARIANTS[variant]
        e = 2 * v["enc"]
        self.block0 = IFBlock(7 + e, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + e, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + e, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + e, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + e, c=v["c4"])
        self.encode = Head(v["enc"])
        self.scale_list = [k / scale for k in v["scales"]]

    @staticmethod
    def _grid(h: int, w: int, device):
        flow_div = torch.tensor([(w - 1.0) / 2.0, (h - 1.0) / 2.0], dtype=torch.float, device=device)
        hor = torch.linspace(-1.0, 1.0, w, dtype=torch.float, device=device).view(1, 1, 1, w).expand(-1, -1, h, -1)
        ver = torch.linspace(-1.0, 1.0, h, dtype=torch.float, device=device).view(1, 1, h, 1).expand(-1, -1, -1, w)
        return flow_div, torch.cat([hor, ver], 1)

    def forward(self, img0, img1, timestep: float, f0=None, f1=None):
        """``img0``/``img1``: (1, 3, H, W) in [0, 1], H/W multiples of
        ``MODULO``. ``timestep`` in (0, 1). ``f0``/``f1`` are optional cached
        ``self.encode`` features (the same frame feeds several timesteps)."""
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        n, _, h, w = img0.shape
        flow_div, grid = self._grid(h, w, img0.device)
        if f0 is None:
            f0 = self.encode(img0)
        if f1 is None:
            f1 = self.encode(img1)
        if torch.is_tensor(timestep):   # traced graphs (Core ML) take it as an input
            ts = timestep.to(img0.dtype).reshape(1, 1, 1, 1).expand(n, 1, h, w)
        else:
            ts = torch.full((n, 1, h, w), float(timestep), dtype=img0.dtype, device=img0.device)

        warped0, warped1 = img0, img1
        flow = mask = feat = None
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        for i, block in enumerate(blocks):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, ts), 1), None, scale=self.scale_list[i])
            else:
                wf0 = _warp(f0, flow[:, :2], flow_div, grid)
                wf1 = _warp(f1, flow[:, 2:4], flow_div, grid)
                fd, mask, feat = block(
                    torch.cat((warped0, warped1, wf0, wf1, ts, mask, feat), 1), flow, scale=self.scale_list[i],
                )
                flow = flow + fd
            warped0 = _warp(img0, flow[:, :2], flow_div, grid)
            warped1 = _warp(img1, flow[:, 2:4], flow_div, grid)
        mask = torch.sigmoid(mask)
        return warped0 * mask + warped1 * (1 - mask)
