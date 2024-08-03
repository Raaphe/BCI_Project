
---


## Running tensors and PyTorch objects on the GPUs (and making faster computations)
---

GPUs = faster computation on numbers, thanks to CUDA + Nvidia hardware + PyTorch working behind the scenes to make everything hunky dunky (good).


```python
import torch;
import argparse;

# Check for GPU access with pytorch
print(torch.cuda.is_available());

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu";

# Count numbers of devices
print(torch.cuda.device_count());


# +==== The code below is from the https://pytorch.org/docs/stable/notes/cuda.html#best-practices website for best practices ====+

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

args = parser.parse_args()
args.device = None

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

// OUTPUT

True
1
```
#### Putting Tensors (and models) on the GPU.
---

The reason we want our tensors/models on the GPU is because using a GPU results in faster computations.

```python
import torch;

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu";

# Create a tensor (default on the cpu).
tensor = torch.tensor([1,2,3])

# Tensor not on GPU
print(F"{tensor} -> {tensor.device}");

# Move tensor to GPU
tensor_on_GPU = tensor.to(device);
print(tensor_on_GPU);

### 4. Moving tensors back to the CPU
# If the tensor is on GPU, can't transorm it to NumPy.
# NumPy only works off of CPU.
# To fix the GPU tensor with NumPy issue, we can first set it to the CPU.
tensor_back_on_cpu = tensor_on_GPU.cpu().numpy();
print(f"NumPy array -> {tensor_back_on_cpu}");
```
