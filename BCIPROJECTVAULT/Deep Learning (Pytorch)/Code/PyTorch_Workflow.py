from torch import nn; # nn contains all of pytorch's building blocks for neural networks
import numpy as np;
import matplotlib.pyplot as plt;
from pathlib import Path;
import torch;

workflow = {
    1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
};

# Check PyTorch version
print(torch.__version__);


# To showcase this, let's create some *known* data using the linear regression formula. 
# We'll use a linear regression formula to make a straight line with known **Parameters**.

# Create *known* parameters
weight = 0.7; # in the formula this is b
bias = 0.3; # and this is a

# Create
start = 0;
end = 1;
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1); # X is a capital because it is a matrix/tensor

y = weight * X + bias


print(y[:10])
print(f"{y.shape}\n")

print(X[:10])
print(f"{X.shape}\n")


### Splitting data into training and tests sets (one of the most important concepts in machine learning in general)

# Create a train/test split
train_split = int(0.8 * len(X));
X_train, y_train = X[:train_split], y[:train_split];
x_test, y_test = X[train_split:], y[train_split:];


print(f"Len of X_train {len(X_train)}")
print(f"Len of y_train {len(X_train)}")
print(f"Len of X_test {len(x_test)}")
print(f"Len of y_test {len(y_test)}")


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=x_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions
    """

    plt.figure(figsize=(10,7));

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="training data");

    # Plot test data in green 
    plt.scatter(test_data, test_labels, c="g", s=4, label="testing data");

    # Are there predictions
    if predictions is not None:
        # Plot predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions");

    # Show the legend
    plt.legend(prop={"size": 17});
    # plt.show();


# Create linear regression model class
class LinearRegressionModel(nn.Module): # Almost everything in pytorch inherits from nn.Module
    def __init__(self):
        super().__init__();
        self.weights = nn.Parameter(torch.randn(1, 
                                                requires_grad=True, # <- Can this parameter be updated via gradient descent?
                                                dtype=torch.float));

        self.bias = nn.Parameter(torch.randn(1, 
                                             requires_grad=True,  # <- Can this parameter be updated via gradient descent?
                                             dtype=torch.float32));

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias




# Create a random seed
RANDOM_SEED = 42;
torch.manual_seed(RANDOM_SEED);

# Create an instance of the model (this is a subclass of nn.Module)
model_0 = LinearRegressionModel();

# Check out parameters
print("\n");
print(model_0);
print(list(model_0.parameters()));

# Check out named parameters (a parameter is a value that the model sets itself)
print(model_0.state_dict());

# Making predictions using `torch.inference_mode()`
with torch.inference_mode():
    y_preds = model_0(x_test);


# Setup a loss function
loss_fn = nn.L1Loss();


# Setup an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.01); # lr = learning rate = possibly the most important hyperparameter you can set


# An epoch is one loop through the data... (this is a hyperparameter because we set it ourself)
epochs = 210;

# Track 
epoch_count = [];
loss_values = [];
test_loss_values = [];

def training():

    # Set the model to training mode.
    model_0.train(); # train mode in PyTorch sets all parameters that require gradients to require gradients

    # 1. Forward Pass on train data using the forward() moethod inside
    y_preds = model_0(X_train);

    # 2. Calculate the loss (how different are the model's predictions to the true values)
    loss = loss_fn(y_preds, y_train );
    loss_values.append(loss);

    # 3. Optimizer zero grad (they accumulate by default)
    optimizer.zero_grad();

    # 4. Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward();

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step();


### Training
# 0. loop through the data
for epoch in range(epochs):
    training();

    ### Testing
    model_0.eval() # Turn off gradient tracking. Turns off settings in the model not needed for evaluation
    with torch.inference_mode(): # Turns off gradient tracking and a couple more things behind the scenes
        # 1. Do the forward pass
        test_pred = model_0(x_test);

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test);
        epoch_count.append(epoch);
        test_loss_values.append(test_loss);
        
        if epoch % 10 == 0 : 

            print(f"Epoch: {epoch} | Test Loss: {test_loss}\n");



plt.plot(epoch_count, np.array(torch.tensor(loss_values).cpu().numpy()), label="Train Loss");
plt.plot(epoch_count, test_loss_values, label="Test Loss");
plt.title("Training and test loss curves");
plt.xlabel("Epochs");
plt.ylabel("Loss");
plt.legend();
# plt.show()

print(model_0.state_dict());
# plot_predictions(predictions=test_pred);
# plt.show();

# Saving our Pytorch model

# 1. Create models directory
MODEL_PATH = Path("models");
MODEL_PATH.mkdir(parents=True, exist_ok=True);

# 2 . Create model save path
MODEL_NAME = "model_01_PyTorch_workflow.pth";
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME;

# 3. Save the model state dict
print(f"saving model to {MODEL_SAVE_PATH}");
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH);

# Loading our Pytorch model

# To load in a saved `state_dict` we have to instantiate a new instance of our model class
loaded_model_0 = LinearRegressionModel();

# Load the saved dict of model_0 (this will update the enw instance with updated Parameters)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH));

# Make some predictions with our loaded model
loaded_model_0.eval();
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(x_test);

model_0.eval();
with torch.inference_mode():
    y_preds = model_0(x_test);

print(y_preds == loaded_model_preds);