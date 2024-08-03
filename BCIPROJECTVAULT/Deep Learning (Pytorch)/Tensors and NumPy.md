
---

NumPy is a popular scientific Python numerical computing library.

And because of this, PyTorch has functionality to interact with it.

* Data in NumPy, want in PyTorch tensor -> `torch.from_numpy(ndarray)`
* PyTorch tensor -> NumPy -> `torch.Tensor.numpy()`

```python
import torch;
import numpy as np;

array = np.arange(1.0, 8.0);
tensor = torch.from_numpy(array); # WARNING : when converting from numpy -> pytorch, pytorch relfects numpy's dtype of float64 to our tensor. (the default in pytorch is of float32)
print(f"numpy array : {array}");
print(f"tensor from array : {tensor}")
print("\n");

# Tensor to NumPy array
print(f"Tensor to numpy")
tensor = torch.ones(7);
numpy_tensor = tensor.numpy();

print(f"pytorch tensor : {tensor}");
print(f"NumPy array from tensor : {numpy_tensor}");
print(f"NumPy array from tensor dtype : {numpy_tensor.dtype}"); # Reflects the tensor's dtype...
```