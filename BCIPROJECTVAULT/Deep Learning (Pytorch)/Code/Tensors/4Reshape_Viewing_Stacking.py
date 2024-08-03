import torch;

## Reshaping, stacking, sqeezing and unsqueezing tensors.

### Reshaping

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


### Changing the view.

x = x.reshape(1,9);
z = x.view(1, 9);
print(f"Tensor : {z}\nShape : {z.shape}\n");

print("x[:,0] = 99");
x[0,0] = 99;
print(f"x : {x}\nz : {z}\n");


### Stack tensors on top of eachother

x = torch.arange(0.,9.);
print(f"Stacked tensors | {x}\n");

x_stacked = torch.stack([x,x,x,x]);
print(f"dim = 0 (default) \n{x_stacked}\n");

x_stacked = torch.stack([x,x,x,x], dim=1);
print(f"dim = 1\n{x_stacked}\n");