import torch


# Helper function for weight quantization
def quantize_weights(weights, num_bits):
    qmin = -2 ** (num_bits - 1)
    qmax = 2 ** (num_bits - 1) - 1
    scale = weights.abs().max() / qmax
    quantized = torch.round(weights / scale).clamp(qmin, qmax) * scale
    return quantized

# Calculate BitOps for a conv layer
def calculate_bitops(conv_layer, input_size, weight_precision):
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size = conv_layer.kernel_size[0] * conv_layer.kernel_size[1]
    stride = conv_layer.stride[0]
    padding = conv_layer.padding[0]

    H_in, W_in = input_size
    H_out = (H_in + 2 * padding - conv_layer.kernel_size[0]) // stride + 1
    W_out = (W_in + 2 * padding - conv_layer.kernel_size[1]) // stride + 1

    ops_per_position = in_channels * kernel_size
    total_ops = ops_per_position * H_out * W_out * out_channels

    return total_ops * weight_precision