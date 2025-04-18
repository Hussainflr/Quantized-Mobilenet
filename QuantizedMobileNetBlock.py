import torch.nn as nn
import torch.nn.functional as F
from utility import quantize_weights, calculate_bitops


# Define a depthwise separable conv block with quantization
class QuantizedMobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, weight_precision):
        super().__init__()
        self.weight_precision = weight_precision
        self.stride = stride

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        self.depthwise.weight.data = quantize_weights(self.depthwise.weight.data, self.weight_precision)
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu6(x)

        self.pointwise.weight.data = quantize_weights(self.pointwise.weight.data, self.weight_precision)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu6(x)
        return x

    def bitops(self, input_size):
        dw_bitops = calculate_bitops(self.depthwise, input_size, self.weight_precision)
        H_out = (input_size[0] + 2 * self.depthwise.padding[0] - 3) // self.stride + 1
        W_out = (input_size[1] + 2 * self.depthwise.padding[0] - 3) // self.stride + 1
        pw_bitops = calculate_bitops(self.pointwise, (H_out, W_out), self.weight_precision)
        return dw_bitops + pw_bitops, (H_out, W_out)