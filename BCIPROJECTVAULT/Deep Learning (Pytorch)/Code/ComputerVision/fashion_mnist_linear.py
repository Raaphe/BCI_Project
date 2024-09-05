import matplotlib.pyplot as plt
import torch;
import torch.utils
from torch.utils.data import DataLoader;
from torch import nn;
import torch.utils.data
import torchvision;
from torchvision import datasets;
DEVICE = "cuda" if torch.cuda.is_available() else "cpu";

class FashionMNISTModel0(nn.Module):

    def __init__(self,
                input_shape:int,
                hidden_units:int,
                output_shape:int):
        super().__init__();

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        );

    def forward(self, x):
        return self.layer_stack(x.to(DEVICE));



def main():

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

    # See first training example
    image :torch.Tensor = train_data[0][0];
    label :int = train_data[0][1]


    class_names = train_data.classes;
    class_to_idx = train_data.class_to_idx;

    print(f"Image shape -> {image.shape} : [Color_channel, height, width] ");
    print(f"Image label : {class_names[label]} ");


    # plt.imshow(image.squeeze(), cmap="gray");
    # plt.title(f"{class_names[label]}");
    # plt.axis(False);


    torch.manual_seed(42);
    fig = plt.figure(figsize=(9,9));
    rows, cols = 4,4;
    for i in range(1, rows*cols+1):

        random_idx = torch.randint(0, len(train_data), size=[1]).item();
        img, label = train_data[random_idx];
        fig.add_subplot(rows, cols, i);
        plt.imshow(img.squeeze(), cmap="gray");
        plt.title(f"{class_names[label]}");
        plt.axis(False);

    # plt.show();


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


    # Show a sample

    train_features_batch, train_labels_batch = next(iter(train_dataloader)); 

    print(f"train_feature_batch shape : {train_features_batch.shape}");
    print(f"train_feature_batch shape : {train_labels_batch.shape}");

    torch.manual_seed(42);
    plt.figure(figsize=(5, 5));
    random_idx = torch.randint(0, len(train_features_batch), size=[1]).item();
    img, label = train_features_batch[random_idx], train_labels_batch[random_idx];
    plt.imshow(img.squeeze(), cmap="gray");
    plt.title(class_names[label]);
    plt.axis(False);

    # Create a baseline model.


    # Create a flatten layer.
    flatten_model = nn.Flatten(); 

    # Get a single sample.
    x = train_features_batch[0]; # Shape = [1, 28, 28];


    # Flatten the sample 
    output = flatten_model(x);

    # Print out whats haapenin
    print(f"shape before flattening {x.shape}");
    print(f"shape after flattening {output.shape}");

    torch.manual_seed(42);

    # Setup model with input parameters
    model = FashionMNISTModel0(
        input_shape=784, # this is 28 x 28 
        hidden_units=10,
        output_shape=len(class_names) # one for every class
    ).to(DEVICE);

    dummy_x = torch.rand([1,1,28,28]).to(DEVICE);
    print(model(dummy_x));

    # Setup loss, optimizer and evaluation metrics.
    #
    # Loss Function - Since we're working with multi-class data, our loss will be `nn.CrossEntrtopyLoss()`
    # Optimizer - out optimizer `torch.optim.SGD()` (stochastic gradient descent)
    # Evaluation Metric - since we're working on a classification problem, let's use accuracy as our evaluation metric.


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


    for epoch in tqdm(range(epochs)):
        
        print(f"Epoch: {epoch}\n------");

        # Training
        train_loss = 0;

        # Add a loop to loop through the training batches
        for batch, (X, y) in tqdm(enumerate(train_dataloader)):
            model.train();
            X, y = X.to(DEVICE), y.to(DEVICE)

            # Forward pass
            y_pred = model(X);
        
            # Calculate the loss
            loss = loss_fn(y_pred, y);
            train_loss += loss; # Accumulate train loss
        
            # Optimizer zero grad
            optimizer.zero_grad();
        
            # loss backward
            loss.backward();
        
            # optimizer step
            optimizer.step();
            
            # Print out what's happening
            if batch % 400 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples");
            
        # Divide total train loss by lenght of train dataloader
        train_loss /= len(train_dataloader);

        # TESTING
        test_loss, test_acc = 0,0;
        model.eval();
        with torch.inference_mode():
            for (X_test, y_test) in test_dataloader:
                X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE);


                # forward pass
                y_test_pred: torch.Tensor = model(X_test);
    
                test_loss += loss_fn(y_test_pred, y_test);
                test_acc = accuracy_fn(y_pred=y_test_pred.argmax(dim=1), y_true=y_test);

            # Calculate the test loss average per batch
            test_loss /= len(test_dataloader);

            # Calculate the test accuracy per batch
            test_acc /= len(test_dataloader);

        # Print out what's happening
        print(f"\nTrain loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc : {test_acc:.4f}%");

    train_time_end = timer();



    total_train_time = print_train_time(start=start_time_train, end=train_time_end, device=str(next(model.parameters()).device));


    # Make predictions and get model resutls

    torch.manual_seed(42);

    def eval_model(model:nn.Module, 
                data_loader: torch.utils.data.DataLoader, 
                loss_fn: nn.Module, 
                accuracy_fn):
        """Returns a dictionnary containing the results of model predicting on data_loader"""
        loss, acc = 0,0;
        model.eval();
        with torch.inference_mode():
            for X,y in tqdm(data_loader):
                X, y = X.to(DEVICE), y.to(DEVICE)

                # make preds
                y_pred = model(X);

                # Accumulate the loss and acc values per batch
                loss += loss_fn(y_pred, y);
                acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1));
        
            # Scale loss and acc to find the average loss/acc per batch
            loss /= len(data_loader);
            acc /= len(data_loader);
        
        return {"model_name": model.__class__.__name__, # Only works when model was created with 
                "model_loss": loss.item(),
                "model_acc": acc
                }

    model_0_results = eval_model(model=model,
                                accuracy_fn=accuracy_fn,
                                data_loader=test_dataloader,
                                loss_fn=loss_fn);
        

    print(model_0_results);

if __name__ == "__main__":
    main();