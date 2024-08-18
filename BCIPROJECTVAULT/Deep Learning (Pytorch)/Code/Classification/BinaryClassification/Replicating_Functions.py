import torch;
import torch.nn as nn;
import matplotlib.pyplot as plt;

# RELU
tensor = torch.arange(-10,10,1);

def relu(x:torch.Tensor) :
    return torch.maximum(torch.tensor(0), x);


plt.figure(figsize=(12,6));
plt.subplot(1,2,1);
plt.title("Our Relu");
plt.plot(relu(tensor));
plt.subplot(1,2,2);
plt.title("Torch Relu");
plt.plot(torch.relu(tensor));



print(tensor)

# SIGMOID


def sigmoid(x:torch.Tensor) :
    return 1 / (1 + torch.exp(-x));

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Our sigmoid")
plt.plot(sigmoid(tensor));
plt.subplot(1,2,2)
plt.title("Torch sigmoid")
plt.plot(torch.sigmoid(tensor));
plt.show();