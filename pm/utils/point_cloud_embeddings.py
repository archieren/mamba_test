# ------------------------------------------------------------------------
# V-DETR
# Copyright (c) V-DETR authors. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from :
# 3DETR
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Group-Free-3D
# Copyright (c) Group-Free-3D authors. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import numpy as np

# ----------------------------------------
# Simple Point manipulations。我将他们改成了torch.tensor版！
# ----------------------------------------
def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3 Tensor
    src_range: [[B x 3], [B x 3]] - List[Tensor] - min and max XYZ coords - 不写成2xBx3的原因，在于最外层是个List！
    dst_range: [[B x 3], [B x 3]] - List[Tensor] - min and max XYZ coords - 不写成2xBx3的原因，在于最外层是个List！
    """
    device=src_range[0].device
    if dst_range is None: #相当于放到0-1体中！
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=device),
            torch.ones((src_range[0].shape[0], 3),  device=device),
        ]

    src_diff = src_range[1]- src_range[0]   # B x 3
    dst_diff = dst_range[1]- dst_range[0]   # B x 3
    scale = dst_diff / src_diff   # Bx3  TODO: 注意分母为零的情况！还未处理！
    prop_xyz = (pred_xyz - src_range[0].unsqueeze(1)) * scale.unsqueeze(1)+ dst_range[0].unsqueeze(1)
    return prop_xyz

class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,          # 输入"位置"的纬度
        gauss_scale=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        """
        Args:
            xyz (tensor): BxNx3
            num_channels (int):
            input_range (List[Tensor]): [Bx3, Bx3]

        Returns:
            _type_: _description_
        """
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                return self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += (
                f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
            )
        return st
    