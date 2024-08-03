
---


Reproducibility is trying to take random out of random.  In short, how a neural network learns:

1. Starts with random numbers
2. tensor operations
3. Update random numbers to try to and make them better representations of the data 
4. again 
5. again 
6. again...

To reduce the randomness in neural networks and PyTorch comes the concept of a **random seed**.
Essentially, what the random seed does is "flavor" the randomness.

> Extra resources for [reproducibility and randomness](https://pytorch.org/docs/stable/notes/randomness.html) and [random seeds](https://en.wikipedia.org/wiki/Random_seed).


```python
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


// OUTPUT
tensor([[0.3701, 0.3905, 0.5315, 0.7483],
        [0.2793, 0.7555, 0.4390, 0.9773],
        [0.6134, 0.9366, 0.1475, 0.4496]])
tensor([[0.5385, 0.7916, 0.1845, 0.6878],
        [0.8848, 0.7490, 0.0893, 0.9301],
        [0.8697, 0.1623, 0.3637, 0.0990]])
tensor([[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]])


tensor([[0.8823, 0.9150, 0.3829, 0.9593],
        [0.3904, 0.6009, 0.2566, 0.7936],
        [0.9408, 0.1332, 0.9346, 0.5936]])
tensor([[0.8823, 0.9150, 0.3829, 0.9593],
        [0.3904, 0.6009, 0.2566, 0.7936],
        [0.9408, 0.1332, 0.9346, 0.5936]])
tensor([[True, True, True, True],
        [True, True, True, True],
        [True, True, True, True]])

```