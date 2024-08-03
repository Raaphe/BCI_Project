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
