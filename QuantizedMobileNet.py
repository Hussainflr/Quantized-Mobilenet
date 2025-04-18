import torch
import torch.nn as nn
import torch.nn.functional as F
from QuantizedMobileNetBlock import QuantizedMobileNetBlock


# Build the full model
class QuantizedMobileNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        input_channels = 3
        for out_channels, stride, bits in config:
            block = QuantizedMobileNetBlock(input_channels, out_channels, stride, bits)
            self.blocks.append(block)
            input_channels = out_channels
        self.classifier = nn.Linear(input_channels, 10)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def total_bitops(self, input_size):
        total = 0
        current_size = input_size
        for block in self.blocks:
            ops, current_size = block.bitops(current_size)
            total += ops
        return total

    def bitops_per_layer(self, input_size):
        layerwise = []
        current_size = input_size
        for idx, block in enumerate(self.blocks):
            ops, current_size = block.bitops(current_size)
            layerwise.append((idx + 1, ops))
        return layerwise