# Gradio app for image and config input
import torch
from torchvision.transforms import Compose, ToTensor, Resize
import gradio as gr
from QuantizedMobileNet import QuantizedMobileNet



def bitops(image, num_blocks, bit_precision, input_size):
    image = image.convert("RGB")
    transform = Compose([Resize((input_size, input_size)), ToTensor()])
    input_tensor = transform(image).unsqueeze(0)

    # Build MobileNet config dynamically
    config = []
    channels = 32
    for i in range(num_blocks):
        stride = 2 if i % 2 == 1 else 1
        config.append((channels, stride, bit_precision))
        channels *= 2

    model = QuantizedMobileNet(config)
    total_bitops = model.total_bitops((input_size, input_size))
    layerwise_bitops = model.bitops_per_layer((input_size, input_size))

    result = f"Results: \n"
    result += f"Estimated Total BitOps: {total_bitops:.2e}\n\n"
    result += "Layer-wise BitOps:\n"
    for idx, ops in layerwise_bitops:
        result += f"  Block {idx}: {ops:.2e} BitOps\n"

    return result

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🧠 Quantized MobileNet BitOps
    Upload an image, choose input size, number of blocks, and bit precision. Get Total and layerwise BitOps stats.
    """)

    with gr.Row():
        image_input = gr.Image(type="pil")
        input_size = gr.Slider(28, 256, value=32, step=4, label="Input Image Size (HxW)")

    with gr.Row():
        num_blocks = gr.Slider(1, 10, value=5, step=1, label="Number of MobileNet Blocks")
        bit_precision = gr.Slider(2, 8, value=4, step=2, label="Bit Precision")

    result = gr.Textbox(label="Output")
    submit = gr.Button("Run Model")

    submit.click(fn=bitops, inputs=[image_input, num_blocks, bit_precision, input_size], outputs=result)

# Example usage
if __name__ == "__main__":
    demo.launch()