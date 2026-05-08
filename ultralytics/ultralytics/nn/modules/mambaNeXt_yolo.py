# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""MambaNeXt-YOLO blocks (Lei et al., arXiv:2506.03654): Stem, VCM, MambaNeXt, MAFPN (MHAF-YOLO topology)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = (
    "AVG",
    "RepHMSMamba",
    "SimpleStem",
    "MambaNeXtBlock",
    "VisionClueMerge",
    "autopad",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 实际内核大小
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动填充
    return p


class LayerNorm2dChannels(nn.Module):
    """LayerNorm over channel dimension for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _ssm_scan_seq(
    a_seq: torch.Tensor,
    b_seq: torch.Tensor,
    delta_seq: torch.Tensor,
) -> torch.Tensor:
    """Paper Eq.(10) along one sequence: h_{t+1} = exp(-Δ_t)⊙h_t + A_t + B_t⊙h_t. Inputs B,L,D."""
    b, ell, d = a_seq.shape
    h_state = a_seq.new_zeros(b, d)
    outs = []
    for t in range(ell):
        alpha = torch.exp(-delta_seq[:, t]) + b_seq[:, t]
        h_state = alpha * h_state + a_seq[:, t]
        outs.append(h_state)
    return torch.stack(outs, dim=1)


def _merge_rm(y: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """B,L,D -> B,D,H,W row-major."""
    return y.transpose(1, 2).contiguous().reshape(y.shape[0], -1, h, w)


def _merge_cm(y: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """B,L,D -> B,D,H,W column-major (from W×H flatten)."""
    return y.transpose(1, 2).contiguous().reshape(y.shape[0], -1, w, h).transpose(2, 3).contiguous()


class _SS2DSelectiveScan(nn.Module):
    """2-D selective scan (SS2D): four complementary 1-D traversals + merge (VMamba / Fig.2 SS2D path).

    Shared input-conditioned projections A,B,Δ (1×1 conv on ``x``), same recurrence as §3.3 Eq.(10)
    applied independently along each scan; outputs are averaged so channels stay ``dim``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj_a = nn.Conv2d(dim, dim, 1, bias=True)
        self.proj_b = nn.Conv2d(dim, dim, 1, bias=True)
        self.proj_delta = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w = x.shape
        a = self.proj_a(x)
        bcoef = self.proj_b(x)
        delta = F.softplus(self.proj_delta(x))

        def rm(xmap: torch.Tensor) -> torch.Tensor:
            return xmap.view(b, d, h * w).transpose(1, 2).contiguous()

        def cm(xmap: torch.Tensor) -> torch.Tensor:
            return xmap.transpose(2, 3).contiguous().view(b, d, h * w).transpose(1, 2).contiguous()

        scans = (
            (lambda m: rm(m), lambda y_seq: _merge_rm(y_seq, h, w)),
            (lambda m: torch.flip(rm(m), dims=[1]), lambda y_seq: _merge_rm(torch.flip(y_seq, dims=[1]), h, w)),
            (lambda m: cm(m), lambda y_seq: _merge_cm(y_seq, h, w)),
            (lambda m: torch.flip(cm(m), dims=[1]), lambda y_seq: _merge_cm(torch.flip(y_seq, dims=[1]), h, w)),
        )

        acc = None
        for to_seq, from_seq in scans:
            sa = to_seq(a)
            sb = to_seq(bcoef)
            sd = to_seq(delta)
            y_seq = _ssm_scan_seq(sa, sb, sd)
            y_map = from_seq(y_seq)
            acc = y_map if acc is None else acc + y_map
        return acc / 4.0


class SimpleStem(nn.Module):
    """Two-stage strided conv stem (patch embedding) as in MambaNeXt-YOLO backbone description."""

    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class VisionClueMerge(nn.Module):
    """Vision Clue Merge (VMamba; Sec. 3.2): chessboard split → 1×1 per quadrant → concat → 1×1 project.

    Achieves 2× spatial reduction (4× area); **no normalization**, following Mamba-YOLO practice.
    """

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        if c2 % 4 != 0:
            raise ValueError(f"VisionClueMerge: c2={c2} must be divisible by 4 (per-quadrant 1×1 output channels).")
        q = c2 // 4
        self.branch = nn.ModuleList(nn.Conv2d(c1, q, 1, bias=False) for _ in range(4))
        self.proj = nn.Conv2d(c2, c2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[..., 0::2, 0::2]
        x1 = x[..., 1::2, 0::2]
        x2 = x[..., 0::2, 1::2]
        x3 = x[..., 1::2, 1::2]
        y = torch.cat([self.branch[i](t) for i, t in enumerate((x0, x1, x2, x3))], dim=1)
        return self.proj(y)


class AVG(nn.Module):
    """Adaptive average pool to fixed relative spatial size (MHAF-YOLO MAFPN helper)."""

    def __init__(self, down_n: int = 2) -> None:
        super().__init__()
        self.down_n = down_n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        oh = max(1, int(h / self.down_n))
        ow = max(1, int(w / self.down_n))
        return F.adaptive_avg_pool2d(x, (oh, ow))


class MambaNeXtBlock(nn.Module):
    """Hybrid CNN + selective SSM block with ResGate fusion (MambaNeXt-YOLO).

    Structure follows Sec. 3.3 / Fig.2: preprocess → ConvNeXt local → **SS2D** global scan → ResGate.
    """

    def __init__(
        self,
        c1: int,
        convnext_dw_kernel: int = 7,
        convnext_ratio: int = 4,
        ssm_expand: float = 2.0,
        ssm_dw_kernel: int = 3,
        resgate_dw_kernel: int = 3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.c1 = c1
        d_inner = max(16, int(c1 * ssm_expand))
        cmid = c1 * convnext_ratio

        # Preprocess: X' = SiLU(BN(Conv1×1(X)))
        self.pre = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )

        # (a) ConvNeXt local (BN after DW; inverted bottleneck with GELU)
        self.local_dw = nn.Conv2d(c1, c1, convnext_dw_kernel, 1, autopad(convnext_dw_kernel), groups=c1, bias=False)
        self.local_bn = nn.BatchNorm2d(c1)
        self.local_pw1 = nn.Conv2d(c1, cmid, 1, bias=False)
        self.local_pw2 = nn.Conv2d(cmid, c1, 1, bias=False)

        self.norm_local = LayerNorm2dChannels(c1, eps=eps)

        # (b) Scan prep: SiLU(DWConv(Linear(F'_local)))
        self.scan_pw = nn.Conv2d(c1, d_inner, 1, bias=False)
        self.scan_dw = nn.Conv2d(
            d_inner,
            d_inner,
            ssm_dw_kernel,
            1,
            autopad(ssm_dw_kernel),
            groups=d_inner,
            bias=False,
        )

        self.ssm = _SS2DSelectiveScan(d_inner)

        self.norm_mamba = LayerNorm2dChannels(d_inner, eps=eps)
        self.out_pw = nn.Conv2d(d_inner, c1, 1, bias=False)

        # (c) ResGate
        self.norm_res = LayerNorm2dChannels(c1, eps=eps)
        self.proj_v = nn.Conv2d(c1, c1, 1, bias=True)
        self.proj_u = nn.Conv2d(c1, c1, 1, bias=True)
        self.gate_dw = nn.Conv2d(c1, c1, resgate_dw_kernel, 1, autopad(resgate_dw_kernel), groups=c1, bias=False)
        self.proj_z = nn.Conv2d(c1, c1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = self.pre(x)

        xdw = self.local_dw(xp)
        flocal = self.local_pw2(F.gelu(self.local_pw1(self.local_bn(xdw))))

        fln = self.norm_local(flocal)
        fscan = F.silu(self.scan_dw(self.scan_pw(fln)))
        fmamba = self.ssm(fscan)
        fmn = self.norm_mamba(fmamba)
        fglobal = self.out_pw(fmn)

        efglobal = fglobal + xp
        fp_global = self.norm_res(efglobal)

        u = self.proj_u(fp_global)
        v = self.proj_v(fp_global)
        z = F.gelu(self.gate_dw(u) + u) * v
        y = fp_global + self.proj_z(z)

        return efglobal + y


class RepHMSMamba(nn.Module):
    """MAFPN fusion block: same branching pattern as MHAF-YOLO ``RepHMS``, inner UniRep branches → ``MambaNeXtBlock``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 3,
        depth: int = 1,
        depth_expansion: int = 2,
        kersize: int = 5,
        shortcut: bool = True,
        expansion: float = 0.5,
        small_kersize: int = 3,
        use_depthwise: bool = True,
    ) -> None:
        super().__init__()
        self.width = width
        self.depth = depth
        _ = (depth_expansion, kersize, shortcut, small_kersize, use_depthwise)
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.branch_blocks = nn.ModuleList()
        for _ in range(width - 1):
            self.branch_blocks.append(nn.ModuleList(MambaNeXtBlock(c_) for _ in range(depth)))
        self.conv2 = Conv(c_ + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x_out = [x[:, i * self.c_ : (i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                if j == self.depth - 1:
                    if self.depth > 1:
                        cascade = [cascade[-1]]
                    else:
                        cascade = []
                x_out[i + 1] = self.branch_blocks[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])
        return self.conv2(torch.cat(elan, 1))
