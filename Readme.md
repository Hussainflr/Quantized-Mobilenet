##You can interact with the app here: [Gradio App](https://huggingface.co/spaces/Hussain5/Quantized-Mobilenet)
## Quantized MobileNet — Variable‑Precision Blocks & BitOps Calculator

This repository demonstrates the design and implementation of a **Convolutional Neural Network (CNN)** architecture using a list of **MobileNet blocks**. The unique aspect of this implementation is that **each block can use a different weight precision**, meaning each layer can have a different number of bits for its weights (e.g., 2-bit, 4-bit, 8-bit). 

The network is designed to process an input image and compute the **BitOps** (bitwise operations) required to process the image through the entire network. This helps in evaluating the computational cost associated with quantized networks and understanding the trade-offs between model size, precision, and computational efficiency.
For any chosen configuration the code:

1. builds the network (depthwise‑separable convolutions),  
2. runs an image through it,  
3. computes **total BitOps** as well as a **layer‑wise BitOps breakdown**, and  
4. serves everything in an interactive **Gradio** app.

The repository grew out of my personal journey to understand **quantization**, **BitOps**, and eventually **Quantization‑Aware Training (QAT)** for ultra‑efficient edge inference.

### 🚀 **What is Quantization and BitOps?**

- **Quantization** is a technique used to reduce the precision of the weights and activations in neural networks, typically to improve inference speed and reduce memory usage. This is particularly useful for deploying models on edge devices with limited computational power and memory.

- **BitOps (Bitwise Operations)**: This refers to the number of bit-level operations performed during model inference. It’s an important metric when evaluating the efficiency of quantized models. Lower bit precision leads to fewer operations and faster inference on resource-constrained devices.

## 🧮 BitOps: The Concept

> **BitOps = Number of operations × Bit-width per operation**

In quantized networks, instead of counting full 32-bit floating point operations (FLOPs), we count operations using lower precision (e.g., 2-bit, 4-bit). So, each multiplication or addition between quantized weights and inputs counts as a **bitwise operation**, scaled by the number of bits.

## 🔍 1. **Depthwise Convolution BitOps**

In a **depthwise convolution**, each input channel is convolved separately:

### **Formula:**

$$
\text{BitOps}_{\text{depthwise}} = H_{\text{out}} \times W_{\text{out}} \times C_{\text{in}} \times K \times K \times B
$$

Where:  
- $H_{\text{out}}, W_{\text{out}}$: height and width of the output feature map  $$
- $C_{\text{in}}$: number of input channels  
- $K \text{x} K$: kernel size (usually \$3 x 3$)  
- $ B$: bit-width used for quantized weights 
 

## 🔍 2. **Pointwise Convolution BitOps**

A **pointwise convolution** is a \(1 \times 1\) convolution across channels, essentially a matrix multiplication.

### **Formula:**

$$
\text{BitOps}_{\text{pointwise}} = H_{\text{out}} \times W_{\text{out}} \times C_{\text{in}} \times C_{\text{out}} \times B
$$

**Where:**

<br>

- $H_{\text{out}},\ W_{\text{out}}$: output size (same as input to pointwise conv)  
- $C_{\text{in}}$: number of input channels  
- $C_{\text{out}}$: number of output channels  
- $B$: bit-width used for quantized weights


- **Output Size**:

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P - K}{S} \right\rfloor + 1
$$

**Where:**

- \( P \): padding  
- \( K \): kernel size  
- \( S \): stride  


## ✅ Example (Depthwise Layer)

For a depthwise conv with:  
- $Input: 32 x 32, 64 channels$  
- $Kernel: 3 x 3$
- $Stride: 1$  
- $Precision: 4-bit$  

$$
\text{BitOps}_{\text{depthwise}} = 32 \times 32 \times 64 \times 3 \times 3 \times 4 = 2,359,296 \text{ BitOps}
$$



---

## 📐  BitOps & Quantization — Why They Matter
| Term | What it means | Why it matters on edge devices |
|------|---------------|--------------------------------|
| **Quantization** | Representing weights/activations with fewer bits (e.g., 8 → 4 → 2). | Cuts model size and memory bandwidth; enables integer‑only arithmetic. |
| **BitOps** | Count of bit‑level operations executed during inference. | Direct proxy for energy/latency on low‑power hardware. |
| **QAT** | Training with fake‑quantized tensors so the model learns to cope with low precision. | Preserves accuracy once the real quantized model is deployed. |

---

## ✨  Key Features
* **MobileNet with Variable Weight Precision**: The MobileNet architecture is implemented with flexibility, where each block can have different weight precision (2-bit, 4-bit, 8-bit).
* **BitOps Calculation**: The implementation computes the total BitOps required for processing an input image through the entire network. This is crucial for assessing the performance of quantized networks on edge devices.
* **Gradio UI** – upload an image and tweak:
  * image size  
  * number of blocks  
  * global bit‑precision  
  * instantly see BitOps.
* **Clean, dependency‑light code** (PyTorch + Gradio).

---

## 🚀  Quick Start – Run the Gradio App

```bash
# 1. Clone
git clone https://github.com/hussainflr/Quantized‑Mobilenet.git
cd Quantized‑Mobilenet

# 2. Install deps
pip install -r requirements.txt   # see list below

# 3. Launch
python app.py
