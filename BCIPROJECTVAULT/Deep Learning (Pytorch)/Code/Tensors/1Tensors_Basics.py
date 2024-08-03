print("\n");
import torch;
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

# Scalar 
print("Scalar num of dimensions - torch.tensor(7)");
scalar = torch.tensor(7);

# Num of Dimensions
print(scalar.ndim);
print("\n");

# Get tensor back as int
print("Get tensor back as int - scalar.item()");
item = scalar.item();
print(item);
print("\n");

# Creating a Vector
print("Creating a Vector, dimension, shape - torch.tensor([7,7])");
vector = torch.tensor([7,7]);
print(vector);
print(vector.ndim);
print(vector.shape);
print("\n");

# MATRIX
print("MATRIX - [[7, 8], [9,10]]");
MATRIX = torch.tensor([
                    [7,8], 
                    [9,10]]);

print(MATRIX);
print(MATRIX.ndim);
print(MATRIX.shape);
print("\n");

# MATRIX MANIPULATION
print("MATRIX - [[7, 8], [9,10]]");
print(MATRIX[0]);
print(MATRIX[1]);
print("\n");


# TENSOR
print("TENSOR - torch.tensor([[ [1,2,3], [4,5,6], [8,9,10] ]]);");
TENSOR = torch.tensor([
    [[1,2,3],
    [4,5,6],
    [8,9,10]]
    ]);

print(TENSOR.ndim);
print(TENSOR.shape);
print("\n");


# Create a random tensor of shape (3,4)
print("Random tensor of size (3,4)");
random_tensor = torch.rand(3, 4);

print(random_tensor);
print(random_tensor.ndim);
print(random_tensor.shape);
print("\n");

# Create a random tensor with similar shape to an image tensor
print("random tensor with similar shape to an image tensor");

random_image_size_tensor = torch.rand(size=(244,244,3)); # height, width, color channels (R, G, B)

print(random_image_size_tensor);
print(random_image_size_tensor.shape);
print(random_image_size_tensor.ndim);
print("\n");

# Create tensors of all zeros or ones

zeros = torch.zeros(size=(3,4));
print(zeros)

ones = torch.ones(size=(3,4));
print(ones)
print(ones.dtype);
print("\n");


# Use torch.arange() 
print("Use torch.arange() ");
range_tensor = torch.arange(start=0, end=1000, step=88);
print(range_tensor);
print("\n");

print("one_to_ten");
one_to_ten = torch.arange(1,10);
print(one_to_ten);
print("\n");

# Creating tensors like 
print("Creating tensors like ");
ten_zeros = torch.zeros_like(input=one_to_ten);
print(ten_zeros);
print("\n");

# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], 
                               dtype=None, # What datatype is the tensor (float32 or float16)
                               device="cuda", # If you try to do operations between two tensors not on the same device, an error will be thrown
                               requires_grad=False); # Whether or not to track gradients with this tensors operations
print(float_32_tensor);
print("\n");


# Messing around
float_16_tensor = float_32_tensor.type(torch.float16);
print(float_16_tensor);
print("\n");

m = float_16_tensor * float_32_tensor;
print(m.dtype);

int_32_tensor = torch.tensor([3,6,9], dtype=torch.long, device="cuda");
print(float_32_tensor * int_32_tensor);

# Getting information from tensors
some_tensor = torch.rand(3,4);

## Find some details about some tensor
print("\n");
print(some_tensor);
print(f"Datatype of tensor: {some_tensor.dtype}");
print(f"Shape of tensor: {some_tensor.shape}");
print(f"Device of tensor: {some_tensor.device}");


