from typing import Tuple;
import matplotlib.pyplot as plt
import torch;
import torch.utils
from torch.utils.data import DataLoader;
from torch import nn;
import torch.utils.data
import torchvision;
from torchvision import datasets;

DEVICE = "cuda" if torch.cuda.is_available() else "cpu";

# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # Do we want the training dataset?
    download=True, # do we want to download?
    transform=torchvision.transforms.ToTensor(), # How do we want to transform the data?
    target_transform=None # How do we want to transform the labels/target?
);

test_data = datasets.FashionMNIST(
    root="data",
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None,
);

class_names = train_data.classes;
class_to_idx = train_data.class_to_idx;


# Setup the batch-size hyper-parameter
BATCH_SIZE = 32;

# Turn datasets into iterables
train_dataloader = DataLoader(
    batch_size= BATCH_SIZE,
    shuffle= True,
    dataset=train_data
);

test_dataloader = DataLoader(
    batch_size= BATCH_SIZE,
    shuffle= False, # no need to shuffle when evaluating
    dataset=test_data
);

print(f"len of train dataloader {len(train_dataloader)}");
print(f"len of test dataloader {len(test_dataloader)}");


train_features_batch, train_labels_batch = next(iter(train_dataloader)); 

# Create a baseline model.



class FashionMNISTModel1(nn.Module):
    def __init__(self,
                 input_shape:int,
                 hidden_units:int,
                 output_shape:int):
        super().__init__();

        self.layer_stack = nn.Sequential(
            nn.Flatten(), # Flatten inputs into a single layer
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        );

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x.to(DEVICE));


torch.manual_seed(42);

# Setup model with input parameters
model = FashionMNISTModel1(
    input_shape=784, # this is 28 x 28 
    hidden_units=10,
    output_shape=len(class_names) # one for every class
).to(DEVICE);

import requests;
from pathlib import Path;
if Path("helper_functions.py").is_file():
    print("already exists");
else:
    print("downloading helper file");
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py");
    with open("helper_functions.py", 'wb') as f:
        f.write(request.content);

from helper_functions import accuracy_fn;

# Setup loss function and optimizer

loss_fn = nn.CrossEntropyLoss();

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1);


from timeit import default_timer as timer;

def print_train_time(start:float, end:float, device: torch.device = None):
    """Prints difference between start and end time."""
    total_time = end-start;
    print(f"Train time on {device}: {total_time:.3f} seconds");
    return total_time;


# Creating a training loop and training a model on batches of data
# 1. Loop through epochs
# 2. Loop through training batches, perform training steps, calculate the train loss *per batch*
# 3. Loop through testing batches, perforrm testing steps, calculate the test loss *per batch*
# 4. Print out what's happening
# 5. Time it all.

from tqdm.auto import tqdm;

torch.manual_seed(42);
start_time_train = timer();

# Set the number of epochs (we'll keep this small for faster training time)
epochs = 1;

def train_step(
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimizer: torch.optim.Optimizer,
    data_loader:torch.utils.data.DataLoader,
    device=DEVICE, 
    ) -> Tuple[int, int]:
    """Performs a training with model trying to learn on data_loader"""
    
    model = model.to(device);
    model.train();
    train_loss, train_acc = 0,0;

    for X, y in data_loader:

        X, y = X.to(device), y.to(device);

        # Forward step
        y_train_preds = model(X);

        # Loss + Acc
        loss = loss_fn(y_train_preds, y);
        train_acc += accuracy_fn(y_true=y, y_pred=y_train_preds.argmax(dim=1));
        train_loss += loss;
    
        # Zero grad
        optimizer.zero_grad();
    
        # back prop
        loss.backward();

        # Step 
        optimizer.step();

    train_loss /= len(data_loader);
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%");
    return tuple([train_loss, train_acc])

def test_step(
    model: nn.Module,
    loss_fn: nn.Module,
    test_data_loader: torch.utils.data.DataLoader,
    device=DEVICE
) -> Tuple[int, int]:

    model = model.to(device);
    model.eval();

    test_acc, test_loss = 0,0;

    with torch.inference_mode():

        for X, y in test_data_loader:

            X, y = X.to(device), y.to(device);

            # Forward pass
            y_test_pred = model(X);
    
            # loss + acc
            loss = loss_fn(y_test_pred, y);
            test_acc += accuracy_fn(y_pred=y_test_pred.argmax(dim=1), y_true=y)
            test_loss += loss;

        test_loss /= len(test_data_loader);
        test_acc /= len(test_data_loader);
        print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n");
    
    return tuple([test_loss, test_acc])
    

# Make predictions and get model resutls
torch.manual_seed(42);

train_time_start_on_gpu = timer();

# Set epochs 
epochs = 3;

# Create a optimization and evalutation loop using `train_step()` and `test_step()`
for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch}");
    print(train_step(
        data_loader=train_dataloader,
        device=DEVICE,
        loss_fn=loss_fn,
        model=model,
        optimizer=optimizer,
    ));

    print(test_step(
        device=DEVICE,
        loss_fn=loss_fn,
        model=model,
        test_data_loader=test_dataloader,
    ));

train_time_end_on_gpu = timer();

total_train_time = print_train_time(start=train_time_start_on_gpu, end=train_time_end_on_gpu, device=str(next(model.parameters()).device));
print(f"=== Time taken to train is: {total_train_time} ===");
