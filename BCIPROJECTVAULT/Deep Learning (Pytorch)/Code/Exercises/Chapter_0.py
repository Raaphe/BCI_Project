import torch;

# 2
random_tensor = torch.randn(7,7);
print(random_tensor);
print(random_tensor.shape);
print("\n");

# 3
print(random_tensor.mm(torch.randn(1,7).T));
print("\n");

# 4 
RANDOM_SEED = 0;
torch.manual_seed(RANDOM_SEED);

## 4-2
random_tensor = torch.randn(7,7);
print(random_tensor);
print(random_tensor.shape);
print("\n");

## 4-3
torch.manual_seed(RANDOM_SEED);
random_tensor2 = torch.randn(1,7).T;
print(random_tensor.mm(random_tensor2));
print(random_tensor == random_tensor2);
print("\n");

# 5 & 6
NEW_SEED = 1234;
DEVICE = "cuda" if torch.cuda.is_available() else "cpu";

torch.cuda.manual_seed(NEW_SEED);
rt1 = torch.randn(2,3);

torch.cuda.manual_seed(NEW_SEED);
rt2 = torch.randn(2,3);

rt1.to(DEVICE);
rt2.to(DEVICE);

# 7
rt1_rt2_product = rt1.T.mm(rt2);
print(rt1);
print(rt2);
print(rt1_rt2_product);
print("\n");

# 8 
print(f"Max: {rt1_rt2_product.max()}");
print(f"Min: {rt1_rt2_product.min()}");
print("\n");

# 9 
print(f"Max Index: {rt1_rt2_product.argmax()}");
print(f"Min Index: {rt1_rt2_product.argmin()}");
print("\n");

# 10

torch.manual_seed(7);
final_random_tensor = torch.randn(1,1,1,10);
vector = final_random_tensor[0][0][0];

print(f"The tensor is \n{final_random_tensor}\n{final_random_tensor.shape}");
print(f"The vector is \n{vector}\n{vector.shape}");
