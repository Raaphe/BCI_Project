import torch;

# Shapes for matrix multiplication

tensor_A = torch.tensor([[1,2],
                         [3,4],
                         [5,6]]);

tensor_B = torch.tensor([[7,10],
                         [8, 11],
                         [9, 12]]);

# torch.mm(tensor_A, tensor_B) # torch.mm is the same as matmul (it's an alias for matmul).

# torch.mm(tensor_A, tensor_B); # This will give an error as 3x2 and 3x2 don't have the correct format for matrix multiplication.

# To fix our tensor shape issues, we can manipulate the shape of one of our tensor using a **transpose**.
# A **transpose** switches the axes or dimensions of a given tensor.

print(tensor_B.shape)

print(tensor_B.T.shape)

print(torch.mm(tensor_B, tensor_A.T).shape);

# Tensor Manipulation
print("\n");

## Create a tensor and add 10 to it
tensor = torch.tensor([1, 2, 3]);
tensor + 10;
print(tensor);

### or

torch.add(tensor, 5);

## Multiply Tensor by 10
tensor = tensor * 10
print(tensor);

### or 
torch.mul(tensor, 10);

## Substract 10
tensor -= 10;
print(tensor);

### or 

torch.sub(tensor, 99)

## Matrix Multiplication
 

### Element-wise Multiplication
print("\n");
print(tensor * tensor);


### Matrix Multiplication

print("\n");
print(f"Equals : ");
print(torch.matmul(tensor, tensor))

