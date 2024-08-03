import torch;

## Finding the min, max, mean, sum etc (tensor aggregation)

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

print(f"index of min : {x.argmin()} | Position of x.argmin() : {x[x.argmin()]}")

print(f"index of max : {x.argmax()} | Position of x.argmax() : {x[x.argmax()]}")


