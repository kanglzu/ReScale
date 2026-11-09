"""Convolution, scale gating and Mamba blocks for RecursiveUNet."""

import torch
from torch import nn


def channel_norm(channels):
    """Use one GroupNorm group per channel, independent of batch size."""
    return nn.GroupNorm(channels, channels)


class ScaleContextBlock(nn.Module):
    """Mix local and wider context features, then expand/compress channels.

    The mixing weight comes from image features and crop scale.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expansion=2,
        context_kernel=7,
        use_context=True,
    ):
        super().__init__()
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if context_kernel < 3 or context_kernel % 2 == 0:
            raise ValueError("context_kernel must be an odd integer >= 3")

        hidden_channels = max(1, int(round(in_channels * expansion)))
        self.local = nn.Conv2d(
            in_channels, in_channels, 3, stride, 1,
            groups=in_channels, bias=False,
        )

        # Context branch: a vertical kernel followed by a horizontal kernel.
        if use_context:
            self.context = nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels, (context_kernel, 1), stride,
                    (context_kernel // 2, 0), groups=in_channels, bias=False,
                ),
                nn.Conv2d(
                    in_channels, in_channels, (1, context_kernel),
                    padding=(0, context_kernel // 2),
                    groups=in_channels, bias=False,
                ),
            )
            gate_channels = max(4, in_channels // 8)
            self.feature_gate = nn.Linear(in_channels, gate_channels)
            self.scale_gate = nn.Linear(2, gate_channels, bias=False)
            self.gate_out = nn.Linear(gate_channels, in_channels)
        else:
            self.context = None

        self.norm = channel_norm(in_channels)
        self.expand = nn.Conv2d(in_channels, hidden_channels, 1)
        self.compress = nn.Conv2d(hidden_channels, out_channels, 1)
        self.activation = nn.GELU()

        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, 1, stride, bias=False
            )

    def forward(self, x, scale=None):
        spatial = self.local(x)

        if self.context is not None:
            context = self.context(x)

            if scale is None:
                scale = x.new_zeros((x.shape[0], 2))

            # Pool [B,C,H,W] features into one [B,C] vector per image.
            pooled = x.mean(dim=(-2, -1))
            feature_signal = self.feature_gate(pooled)
            scale = scale.to(dtype=pooled.dtype)
            scale_signal = self.scale_gate(scale)

            gate = feature_signal + scale_signal
            gate = self.activation(gate)
            gate = self.gate_out(gate)
            gate = gate.sigmoid()

            # Convert [B,C] weights to [B,C,1,1] for spatial broadcasting.
            gate = gate.unsqueeze(2)
            gate = gate.unsqueeze(3)

            spatial = spatial + gate * (context - spatial)

        # Channel mixing: normalize -> expand -> activate -> compress.
        y = self.norm(spatial)
        y = self.expand(y)
        y = self.activation(y)
        y = self.compress(y)
        residual = self.shortcut(x)
        return residual + y


class BottleneckMamba(nn.Module):
    """Compress bottleneck channels, run Mamba, then restore the channels."""

    def __init__(self, channels, mamba_channels=64, d_state=16):
        super().__init__()
        if mamba_channels < 1 or d_state < 1:
            raise ValueError("mamba_channels and d_state must be positive")

        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "ReScale requires mamba_ssm for its compressed bottleneck."
            ) from exc

        width = min(channels, int(mamba_channels))
        self.reduce = nn.Conv2d(channels, width, 1)
        self.norm = nn.LayerNorm(width)
        self.mamba = Mamba(d_model=width, d_state=int(d_state), d_conv=4, expand=2)
        self.restore = nn.Conv2d(width, channels, 1)
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # Run this small block in FP32, then restore the outer AMP dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            reduced = self.reduce(x.float())
            batch, channels, height, width = reduced.shape

            # Convert the feature map [B,C,H,W] to spatial tokens [B,H*W,C].
            sequence = reduced.flatten(2)
            sequence = sequence.transpose(1, 2)
            sequence = self.norm(sequence)
            mixed = self.mamba(sequence)

            # Convert tokens back to the bottleneck feature map.
            mixed = mixed.transpose(1, 2)
            mixed = mixed.reshape(batch, channels, height, width)
            correction = self.restore(mixed)
            correction = correction * self.strength.tanh()

        correction = correction.to(dtype=x.dtype)
        return x + correction
