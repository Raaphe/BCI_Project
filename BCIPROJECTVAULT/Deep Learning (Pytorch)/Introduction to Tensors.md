
---

## Creating Tensors
---

PyTorch tensors are created using `torch.Tensor()`. The docs can be found [here](httpss://pytorch.org/docs/stable/tensors.html).

```python
# Scalar
scalar = torch.tensor(7);
print(scalar);

// OUTPUT
> tensor(7)
```

### Some Properties 
---

##### Number of Dimensions
```python
# Num of Dimensions
print(scalar.ndim);

// OUTPUT
> 0
```

##### Tensor as int
```python
# Get tensor back as int
item = scalar.item();
print(item);

// OUTPUT
> 7
```

##### Creating a Vector
```python
# Creating a Vector
vector = torch.tensor([7,7]);
print(vector);

# Dimensions = rows
print(vector.ndim);

# Shape = volume or num of el -> columns
print(vector.shape);

// OUTPUT
> tensor([7, 7])
> 1
> torch.Size([2])
```

##### Matrices
```python
# MATRIX
MATRIX = torch.tensor([
                    [7,8],
                    [9,10]]);

  

print(MATRIX);
print(MATRIX.ndim);
print(MATRIX.shape);

// OUTPUT 
> tensor([[ 7,  8],
        [ 9, 10]]);
> 2
> torch.Size([2,2]);
```

###### Accessing Matrix Values
```python
# MATRIX MANIPULATION
print(MATRIX[0]);
print(MATRIX[1]);

// OUTPUT
> tensor([7, 8])
> tensor([9, 10]
```

##### Tensors
```python
# TENSOR
TENSOR = torch.tensor([

    [[1,2,3],

    [4,5,6],

    [8,9,10]]

    ]);
    
print(TENSOR.ndim);
print(TENSOR.shape);

// OUTPUT
> 3
> torch.Size([1, 3, 3])
```

![[Pasted image 20240723174301.png]]


| **Name**   | What is it?                                                                                   | # of Dimensions                                                                        | lower or upper (usually/example) |
| ---------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------- |
| **scalar** | a single number                                                                               | 0                                                                                      | `Lower(a)`                       |
| **vector** | a number with direction (e.g. wind speed with direction) but can also have many other numbers | 1                                                                                      | `Lower(y)`                       |
| **matrix** | a 2-dimensional array of numbers                                                              | 2                                                                                      | `Upper(Q)`                       |
| **tensor** | a n-dimensional array of numbers                                                              | can be any number, a 0-dimension tensor is a scalar, a 1-dimension tensor is a vector. | `Upper(X)`                       |

## Random Tensors
---

Why random tensors? 

Random tensors are important because the way many neural networks learn is that they start with tensors full of random numbers and then adjust those random numbers to better represent data

`Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers`

Torch random Tensors [here](https://pytorch.org/docs/stable/generated/torch.rand.html)

```python
# Create a random tensor of shape (3,4)
random_tensor = torch.rand(3, 4);

print(random_tensor);
print(random_tensor.ndim);
print(random_tensor.shape);

// OUTPUT
> tensor([[0.8330, 0.7515, 0.0742, 0.3301],
        [0.8422, 0.7147, 0.7002, 0.1055],
        [0.9751, 0.3860, 0.3071, 0.0611]])
> 2
> torch.size([3,4])
```

### Example use case
---

For example, you could represent an image as a tensor with shape `[3, 224, 224]` which would mean `[colour_channels, height, width]`, as in the image has `3` color channels (red, green, blue), a height of `224` pixels and a width of `224` pixels.

![[Pasted image 20240724104450.png]]```


```python
# Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(244,244,3)); # height, width, color channels (R, G, B)

print(random_image_size_tensor.shape);
print(random_image_size_tensor.ndim);

// OUTPUT
> torch.Size([244, 244, 3])
> 3


```

## Zeros  and Ones
---

Create a tensor of all zeros :

```python

# Create tensors of all zeros or ones
zeros = torch.zeros(size=(3,4));
print(zeros)

ones = torch.ones(size=(3,4));
print(ones)

// OUTPUT

> tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
> tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])

```

> Unless you specifically specify what the default datatype of your tensors they will always be `torch.float32`

```python
print(ones.dtype);

// OUTPUT 

> torch.float32
```

## Creating a range of tensors and tensors-like
---

```python

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

// OUTPUT

> Use torch.arange()
> tensor([  0,  88, 176, 264, 352, 440, 528, 616, 704, 792, 880, 968])


> one_to_ten
> tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])


> Creating tensors like
> tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
```


## Tensor Datatypes
---

The possible datatypes for tensors can be found [here](https://pytorch.org/docs/stable/tensors.html).

> **Notes**: tensor datatype is one of the 3 big errors you'll run into with PyTorch & Deep Learning.
> 1. Tensors not right datatype.
> 2. Tensors not right shape.
> 3. Tensors not on the right device. 

```python
# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # What datatype is the tensor (float32 or float16)
                               device="cuda", # If you try to do operations between two tensors not on the same device, an error will be thrown
                               requires_grad=False); # Whether or not to track gradients with this tensors operations.
                               
float_16_tensor = float_32_tensor.type(torch.float16);

print(float_32_tensor);
print(float_16_tensor);

// OUTPUT

> tensor([3., 6., 9.], device='cuda:0')
> tensor([3., 6., 9.], device='cuda:0', dtype=torch.float16)
```

## Getting information from Tensors (Tensor Attributes)
---

> 1. Tensors not right datatype. - to do get datatype from a tensor, can use `tensor.dtype`
> 2. Tensors not right shape. - to get shape from a tensor, can use `tensor.shape`
> 3. Tensors not on the right device. - to get device from a tensor, can use `tensor.device`

```python
## Find some details about some tensor
print(some_tensor);
print(f"Datatype of tensor: {some_tensor.dtype}");
print(f"Shape of tensor: {some_tensor.shape}");
print(f"Device of tensor: {some_tensor.device}");
```

## Manipulating Tensors (tensor operations)
---

> Tensor operations include: 
> - Addition
> - Subtraction (element-wise)
> - Multiplication
> - Division
> - Matrix Multiplication

### Addition
```python
## Create a tensor and add 10 to it
tensor = torch.tensor([1, 2, 3]);
tensor + 10;
print(tensor);

### or

torch.add(tensor, 5);
print(tensor);
```

### Substraction
```python
## Substract 10
tensor -= 10;
print(tensor);

### or

torch.sub(tensor, 99)
```

### Scalar Multiplication
```python
## Multiply Tensor by 10
tensor = tensor * 10
print(tensor);

### or

torch.mul(tensor, 10);
```


### Matrix Multiplication
---
Two main ways of performing multiplication in neural networks and deep learning:

1. Element-wise multiplication
2. Matrix Multiplication (dot product)

There are two main rules that performing matrix multiplication needs to satisfy:

1. The **inner dimensions** must match:
	* `(3, 2) @ (3, 2)` won't work
	* `(2, 3) @ (3, 2)` will work
	* `(3, 2) @ (2, 3)` will work
2. The resulting matrix has the shape of the **outer dimensions**:
	- `(2, 3) @ (3, 2)`  -> `(2, 2)`
	- `(3, 2) @ (2, 3)`  -> `(3, 3)`
#### Element Wise
---

Element-wise multiplication, denoted as `tensor * tensor`, performs the multiplication operation independently on each corresponding pair of elements in the two tensors. This means that the shape of both tensors must either be the same, or they must be broadcastable to a common shape according to PyTorch's broadcasting rules.

For example, if you have two tensors `A` and `B` of shapes `(n,)` and `(m,)` respectively, `A * B` would result in a tensor of shape `(max(n, m),)` assuming broadcasting is possible. Each element in the resulting tensor is computed as `A[i] * B[j]` for each valid index `i` and `j`.

```python
# Example tensors 
A = torch.tensor([1, 2, 3]); 
B = torch.tensor([4, 5, 6]);

# Element-wise multiplication 
result = A * B; 

// OUTPUT

> tensor([ 4, 10, 18])
```

#### Matrix Multiplication
---

Matrix multiplication, performed with `torch.matmul(tensor, tensor)`, treats the input tensors as matrices (or higher-dimensional analogues) and computes the product according to linear algebra rules. Specifically, for 2D tensors (matrices), `torch.matmul(A, B)` computes the matrix product of `A` and `B`, where the inner dimensions must match (i.e., if `A` is `(n, m)` and `B` is `(m, p)`, the result will be `(n, p)`).

For higher-dimensional inputs, `torch.matmul` behaves similarly but operates over the last two dimensions of the input tensors. This makes `torch.matmul` versatile for performing batched matrix multiplications efficiently.

```python
import torch

# Example matrices
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# Matrix multiplication
result = torch.matmul(A, B)

// OUTPUT

> tensor([[19, 22],
         [43, 50]])

```

### Finding the min, max, mean, sum etc (tensor aggregation)
---

```python
### Create a tensor

print("\n");
x = torch.arange(0, 100, 10, dtype=torch.float32);
print(f"{x}\n");

### Find the min
print(f"Min (torch.min(x)): {torch.min(x)}");
print(f"Min (x.min()): {x.min()}");
print("\n");

### Find the max
print(f"Max (torch.max(x)): {torch.max(x)}");
print(f"Max (x.max()): {x.max()}");
print("\n");
  
### Find the mean
print(f"Mean (torch.mean(x)): {torch.mean(x)}");
print(f"Mean (x.mean()): {x.mean()}");
print("\n");

### Find the Sum
print(f"Sum (torch.sum(x)): {torch.sum(x)}");
print(f"Sum (x.sum()): {x.sum()}");
print("\n");

print(f"index of min : {x.argmin()} | Position of x.argmin() : {x[x.argmin()]}");
print(f"index of max : {x.argmax()} | Position of x.argmax() : {x[x.argmax()]}");

// OUTPUT

> tensor([ 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.])

> Min (torch.min(x)): 0.0
> Min (x.min()): 0.0

> Max (torch.max(x)): 90.0
> Max (x.max()): 90.0

> Mean (torch.mean(x)): 45.0
> Mean (x.mean()): 45.0

> Sum (torch.sum(x)): 450.0
> Sum (x.sum()): 450.0

> index of min : 0 | Position of x.argmin() : 0.0
> index of max : 9 | Position of x.argmax() : 90.0
```

### Reshaping, stacking, squeezing and unsqueezing tensors
---

* Reshaping - reshapes an input tensor to a defined shape
* View - Return a view of an input tensor of a certain shape but keep the same memory as the original tensor
* Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
* Squeeze - removes all `1` dimensions from a tensor
* Unsqueeze - add a `1` dimension to a target tensor
* Permute - Return a view of the input  with dimensions permuted (swapped) in a certain way.

#### Reshaping Tensors
---

> This section will cover the method `Tensor.reshape()`.

Lets say we have a tensor like so : 
$$tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])$$
```python
x = torch.arange(1.,10.);

print(f"Tensor: {x}\nShape : {x.shape}\n");
```


We can change the dimensions of the tensor like so:

```python
import torch;

## Reshaping, stacking, sqeezing and unsqueezing tensors.
x = torch.arange(1.,10.);
print(f"Tensor: {x}\nShape : {x.shape}\n");

print("Add an extra dimension (x.reshaped)");
x_reshaped = x.reshape(1, 9);
print(f"Tensor: {x_reshaped}\nShape : {x_reshaped.shape}\n");

print("Add an extra dimension and flip it,");
x_reshaped = x.reshape(9, 1);
print(f"Tensor: {x_reshaped}\nShape : {x_reshaped.shape}\n");

print("Honorable mention...");
x_reshaped = x.reshape(3, 3);
print(f"Tensor: {x_reshaped}\nShape : {x_reshaped.shape}\n");

// OUTPUT

Tensor: tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.])
Shape : torch.Size([9])

Add an extra dimension (x.reshaped)
Tensor: tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.]])
Shape : torch.Size([1, 9])

Add an extra dimension and flip it,
Tensor: tensor([[1.],
        [2.],
        [3.],
        [4.],
        [5.],
        [6.],
        [7.],
        [8.],
        [9.]])
Shape : torch.Size([9, 1])

Honorable mention...
Tensor: tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]])
Shape : torch.Size([3, 3])
```

#### Viewing tensors
---

From my understanding this is kind of like database views where you can make a query of a database and make a compound "table" from that query. Making a view of a tensor is kind of like this where you can reshape your tensor to your liking and if the original tensor changes, the view will follow. eg;

$$tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])$$

```python
### Changing the view.
x = x.reshape(1,9);
z = x.view(1, 9);
print(f"Tensor : {z}\nShape : {z.shape}\n");

print("x[0,0] = 99");
x[0,0] = 99;
print(f"x : {x}\nz : {z}\n");

// OUTPUT

Tensor : tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.]])
Shape : torch.Size([1, 9])

x[0,0] = 99
x : tensor([[99.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]])
z : tensor([[99.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]])
```


#### Stacking tensors
---

```python
  

### Stack tensors on top of eachother

x = torch.arange(0.,9.);
print(f"Stacked tensors | {x}\n");

x_stacked = torch.stack([x,x,x,x]);
print(f"dim = 0 (default) \n{x_stacked}\n");

x_stacked = torch.stack([x,x,x,x], dim=1);
print(f"dim = 1\n{x_stacked}\n");
```

#### Squeezing, Un-squeezing and Permuting Tensors
---

```python
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

// OUTPUT

tensor([0., 1., 2., 3., 4., 5., 6., 7., 8.])
tensor([[0., 1., 2., 3., 4., 5., 6., 7., 8.]])


Previous tensor : tensor([[0., 1., 2., 3., 4., 5., 6., 7., 8.]])
Previous shape : torch.Size([1, 9])


x_reshaped squeezed :
tensor([0., 1., 2., 3., 4., 5., 6., 7., 8.])


Previous target : tensor([0., 1., 2., 3., 4., 5., 6., 7., 8.])
Previous target's Shape: torch.Size([9])


x_squeezed un-squeezed will be :
New tensor: tensor([[0., 1., 2., 3., 4., 5., 6., 7., 8.]])
New Shape: torch.Size([1, 9])


Change the color channels value to become the first dimesions -> [3, 224, 224]
Previous Shape : torch.Size([224, 224, 3])
New Shape : torch.Size([3, 224, 224])
```

### Indexing (selecting data from tensors)
---

indexing with PyTorch is similar to indexing in NumPy. 

```python
import torch;

# Create a tensor
x = torch.arange(1,10).reshape(1,3,3);
  
print(f"x : {x}");
print(f"x shape: {x.shape}");
print("\n");

  
# Indexing out new tensor
print(f"x[0] : \n{x[0]}");
print("\n");

# Indexing on the middle layer (dim=1)
print(f"x[0][0] : \n{x[0][0]}");
print("\n");

# Indexing on the last layer (dim=2)
print(f"x[0][0][0b ] : \n{x[0][0][0]}");
print("\n");

# You can also use the `:` operator to select "all" of a target dimension.
print(f"x[:,0] : \n{x[:,0]}")
print("\n");

# Get all values of the 0th dimension but only index 1 of the second dimension
print(f"x[:,:,1] : \n{x[:,:,1]}")
print("\n");

  
# Get all values of the 0th dimension but only the 1 index value of the 1st and 2nd dimension
print(f"x[:,1,1] : \n{x[:,1,1]}")
print("\n");

# Get index 0 of both 0th and 1st dimension and all values of the 2nd dimension
print(f"x[0,0,:] : \n{x[0,0,:]}")
print("\n");

  
# Index on x to return 9
print(f"x[0][2][2] : \n{x[:,2,2]}")
print("\n");
  
# Index on x to return [3,6,9]
print(f"x[:,:,2] : \n{x[:,:,2][0]}")
print("\n");
```