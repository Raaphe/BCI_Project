import torch;

# Squeezing, Un-squeezing and Permuting Tensors

## Setup
x = torch.arange(0.,9.);
x_reshaped = x.reshape(1,9)

print("\n");
print(x);
print(x_reshaped);
print("\n");

# torch.squeeze() - removes all single dimensions from a target tensor.
print(f"Previous tensor : {x_reshaped}");
print(f"Previous shape : {x_reshaped.shape}");
print("\n")

# Remove extra dimesions from x_reshaped
x_squeezed = x_reshaped.squeeze();
print(f"x_reshaped squeezed :\n{x_squeezed}");
print("\n")

# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim.
print(f"Previous target : {x_squeezed}")
print(f"Previous target's Shape: {x_squeezed.shape}")
print("\n")

# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0);
print(f"x_squeezed un-squeezed will be :")
print(f"New tensor: {x_unsqueezed}");
print(f"New Shape: {x_unsqueezed.shape}");
print("\n")

# torch.permute() - Returns a view of the original tensor `input` with its dimensions permuted
# In other words, rearranges the dimensions of a target tensor in a specified order while also having it as a view variable.

x_original = torch.randn(size=(224,224,3)); # [height, width, color_channels]


# Permute the original tensor to rearrange the axis (or dim) order.
x_permuted = x_original.permute(2, 0, 1); # shifts axis 0->1, 1->2, 2->0;
print(f"Change the color channels value to become the first dimesions -> [3, 224, 224]");
print(f"Previous Shape : {x_original.shape}");
print(f"New Shape : {x_permuted.shape}")


