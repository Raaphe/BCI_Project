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



class FashionMNISTModel2(nn.Module):
    """
    Model architecture that replicates the TinyVGG
    model from cnn explainer website.
    """
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int, device: torch.device = DEVICE):
        super().__init__();

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape, 
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units, 
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        );
    
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units, 
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units, 
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        );

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, # we first have the input shape of the image which is 28*28 which then gets maxpooled to 14*14 then to 7*7. The tensor shape that we get from conv_block_2 is that of 1*10*7*7..
                    out_features=output_shape)
        );


    def forward(self, x: torch.Tensor): 
        x = self.conv_block_1(x);
        # print(f"Output shape of conv_block_1: {x.shape}");
        x = self.conv_block_2(x);
        # print(f"Output shape of conv_block_2: {x.shape}");
        x = self.classifier(x);
        # print(f"Output shape of classifier: {x.shape}");
        return x;
        # return self.conv_block_1(self.conv_block_2(self.classifier(x))); # I tried this but it may not work

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

def train_step(
    model: nn.Module, 
    loss_fn: nn.Module, 
    optimizer: torch.optim.Optimizer,
    data_loader:torch.utils.data.DataLoader,
    device=DEVICE, 
    ) -> Tuple[int, int]:
    """Performs a training step with model trying to learn on data_loader"""
    
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
    """Performs a testing step with model to evaluate."""

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
torch.cuda.manual_seed(42);

# Setup model with input parameters
model = FashionMNISTModel2(
    input_shape=1, # number of channels (e.g. 1 for grayscale, 3 for rgb)
    hidden_units=30,
    output_shape=len(class_names) # one for every class
).to(DEVICE);


# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss();
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1);


train_time_start_on_gpu = timer();

# Set epochs 
epochs = 4;

# Create a optimization and evalutation loop using `train_step()` and `test_step()`
for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch}----\n");
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

from pathlib import Path;
Path("models").mkdir(parents=True, exist_ok=True);
MODEL_NAME = "mnist_model.pth"
MODEL_SAVE_PATH = "models" / MODEL_NAME;
if Path(MODEL_SAVE_PATH).is_file():
    0;
else:
    torch.save(
        obj=model.state_dict(), 
        f=MODEL_SAVE_PATH
    );