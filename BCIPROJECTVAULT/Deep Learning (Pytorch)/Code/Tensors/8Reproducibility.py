import torch;

# Create two random tensors

random_a = torch.rand(3,4);
random_b = torch.rand(3,4);

print(random_a);
print(random_b);
print(random_b == random_a);
print("\n");

# Let's make some random but reproducible tensors

# Set the random seed
RANDOM_SEED = 42;

torch.manual_seed(42);
random_c = torch.rand(3,4);

torch.manual_seed(42);
random_d = torch.rand(3,4);

print(random_c);
print(random_d);
print(random_c == random_d);