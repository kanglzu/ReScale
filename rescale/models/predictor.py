"""The shared U-shaped segmentation model used in both rounds."""
from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from rescale.models.blocks import BottleneckMamba, ScaleContextBlock, channel_norm

class _Stage(nn.Module):
    """Run the convolution blocks of one encoder or decoder stage."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, scale):
        for block in self.blocks:
            x = block(x, scale)
        return x

class RecursiveUNet(nn.Module):
    """Five-level ReScale predictor shared across global and focused views.

    scale is [B, 2]: log(crop width / image width) and
    log(crop height / image height). A full image uses [0, 0].

    Training with deep supervision returns five predictions, largest first.
    Evaluation returns only the full-resolution prediction.
    """

    def __init__(self, in_channels=3, num_classes=2, channels=(32, 64, 128, 256, 512), blocks_per_stage=2, deep_supervision=False, innovation=None):
        super().__init__()
        stage_widths = []
        for width in channels:
            stage_widths.append(int(width))
        self.channels = tuple(stage_widths)
        if len(self.channels) != 5:
            raise ValueError('channels must contain five positive stage widths')
        for width in self.channels:
            if width < 1:
                raise ValueError('channels must contain five positive stage widths')
        if blocks_per_stage < 1 or in_channels < 1 or num_classes < 1:
            raise ValueError('channel counts and blocks_per_stage must be positive')
        if innovation is None:
            options = {}
        else:
            options = dict(innovation)
        self.deep_supervision = bool(deep_supervision)
        self.num_classes = int(num_classes)
        self.blocks_per_stage = int(blocks_per_stage)
        self.context_kernel = int(options.get('context_kernel', 7))
        expansion = options.get('expansion', 2)
        self.expansions = []
        if isinstance(expansion, Sequence) and (not isinstance(expansion, (str, bytes))):
            if len(expansion) != 5:
                raise ValueError('a stage-wise expansion must have five values')
            for value in expansion:
                self.expansions.append(float(value))
        else:
            for level in range(5):
                self.expansions.append(float(expansion))
        self.stem = nn.Sequential(nn.Conv2d(in_channels, self.channels[0], 3, padding=1, bias=False), channel_norm(self.channels[0]), nn.GELU())
        self.encoder = nn.ModuleList()
        for level in range(4):
            if level == 0:
                input_channels = self.channels[0]
                stride = 1
            else:
                input_channels = self.channels[level - 1]
                stride = 2
            stage = self._make_stage(input_channels, self.channels[level], level, stride)
            self.encoder.append(stage)
        self.bottleneck = self._make_stage(self.channels[3], self.channels[4], 4, 2)
        self.bottleneck_context = BottleneckMamba(
            self.channels[4], int(options.get('mamba_channels', 64)), int(options.get('d_state', 16)))
        self.up_projections = nn.ModuleDict()
        for level in range(4):
            self.up_projections[str(level)] = nn.Conv2d(self.channels[level + 1], self.channels[level], 1)
        self.decoder = nn.ModuleDict()
        for level in range(4):
            self.decoder[str(level)] = self._make_stage(self.channels[level], self.channels[level], level, 1)
        self.heads = nn.ModuleDict()
        for level in range(5):
            if level == 0 or self.deep_supervision:
                self.heads[str(level)] = nn.Conv2d(self.channels[level], num_classes, 1)

    def _make_stage(self, in_channels, out_channels, level, stride):
        """The first block changes resolution/channels; later blocks keep them."""
        blocks = []
        for block_index in range(self.blocks_per_stage):
            if block_index == 0:
                block_in_channels = in_channels
                block_stride = stride
            else:
                block_in_channels = out_channels
                block_stride = 1
            block = ScaleContextBlock(block_in_channels, out_channels, block_stride,
                                      expansion=self.expansions[level], context_kernel=self.context_kernel,
                                      use_context=level >= 1)
            blocks.append(block)
        return _Stage(blocks)

    def _decode_stage(self, x, skip, level, scale):
        """One decoder step: project -> resize -> merge the skip -> refine."""
        key = str(level)
        up = self.up_projections[key](x)
        up = F.interpolate(up, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = up + skip
        x = self.decoder[key](x, scale)
        return x

    @property
    def main_head(self):
        return self.heads['0']

    def forward(self, images, scale=None):
        if images.ndim != 4:
            raise ValueError('images must have shape [B,C,H,W]')
        if scale is None:
            scale = images.new_zeros((images.shape[0], 2))
        elif scale.shape != (images.shape[0], 2):
            raise ValueError('scale must have shape [B,2]')
        scale = scale.to(device=images.device, dtype=images.dtype)
        supervise = self.training and self.deep_supervision
        stem = self.stem(images)
        encoder0 = self.encoder[0](stem, scale)
        del stem
        encoder1 = self.encoder[1](encoder0, scale)
        encoder2 = self.encoder[2](encoder1, scale)
        encoder3 = self.encoder[3](encoder2, scale)
        bottleneck = self.bottleneck(encoder3, scale)
        bottleneck = self.bottleneck_context(bottleneck)
        logits4 = None
        if supervise:
            logits4 = self.heads['4'](bottleneck)
        decoder3 = self._decode_stage(bottleneck, encoder3, 3, scale)
        del bottleneck, encoder3
        logits3 = None
        if supervise:
            logits3 = self.heads['3'](decoder3)
        decoder2 = self._decode_stage(decoder3, encoder2, 2, scale)
        del decoder3, encoder2
        logits2 = None
        if supervise:
            logits2 = self.heads['2'](decoder2)
        decoder1 = self._decode_stage(decoder2, encoder1, 1, scale)
        del decoder2, encoder1
        logits1 = None
        if supervise:
            logits1 = self.heads['1'](decoder1)
        decoder0 = self._decode_stage(decoder1, encoder0, 0, scale)
        del decoder1, encoder0
        logits0 = self.heads['0'](decoder0)
        if supervise:
            return [logits0, logits1, logits2, logits3, logits4]
        return logits0
