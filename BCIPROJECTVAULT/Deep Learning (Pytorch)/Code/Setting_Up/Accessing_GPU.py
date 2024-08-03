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

