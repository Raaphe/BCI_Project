from pathlib import Path
import torch;
from torch import nn;
import matplotlib.pyplot as plt;

def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions=None):
    """
    Plots training data, test data and compares predictions
    """

    plt.figure(figsize=(10,7));

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=10, label="training data");

    # Plot test data in green 
    plt.scatter(test_data, test_labels, c="g", s=10, label="testing data");

    # Are there predictions
    if predictions is not None:
        # Plot predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=10, label="Predictions");

    # Show the legend
    plt.legend(prop={"size": 17});
    # plt.show();


DEVICE = "cuda" if torch.cuda.is_available() else "cpu";
print(f"we are using device : {DEVICE}");


# Create some data using the linear regression formula of y = weight * X + bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1);
y = weight * X + bias;

# Split data

train_split = int(0.8 * len(X));
X_train, y_train = X[:train_split], y[:train_split];
X_test, y_test = X[train_split:], y[train_split:];

# Plot the data
# plot_predictions(X_train, y_train, X_test, y_test);
# plt.show();

# Create a Linear Model by subclassing nn.Module
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__();

        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, out_features=1);

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x); 


# Set the manual seed
torch.manual_seed(42);
model_1 = LinearRegressionModelV2();
print(model_1.state_dict())

model_1.to(DEVICE);

### 6.3 Training

# Setup loss function
loss_fn = nn.L1Loss();

# Optimizers
optimizer = torch.optim.SGD(params=model_1.parameters(), 
                            lr=0.01);

torch.manual_seed(42);

epochs = 200;

# Put data on the same device
X_test = X_test.to(DEVICE);
X_train = X_train.to(DEVICE);
y_test = y_test.to(DEVICE);
y_train = y_train.to(DEVICE);

for epoch in range(epochs):

    model_1.train();

    # 1. Forward
    y_pred = model_1(X_train);

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train);

    # 3. optimizer 
    optimizer.zero_grad();

    # 4. backward prop
    loss.backward();

    # 5. Step
    optimizer.step();

    ### Testing

    model_1.eval();
    with torch.inference_mode():
        test_pred = model_1(X_test);
        test_loss = loss_fn(test_pred, y_test);

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss {loss} | Test Loss {test_loss}")

model_1.eval();
with torch.inference_mode():
    y_pred = model_1(X_test);
    # plot_predictions(predictions=y_pred.cpu(), test_data=X_test.cpu(), train_data=X_train.cpu(), test_labels=y_test.cpu(), train_labels=y_train.cpu());
    
# 2 . Create model save path
MODEL_PATH = Path("models");
MODEL_NAME = "model_02_PyTorch_workflow.pth";
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME;

# Saving model
torch.save(obj=model_1.state_dict(),f=MODEL_SAVE_PATH);

# Load model
loaded_model_1 = LinearRegressionModelV2();

loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH));

# Put the target model on device
loaded_model_1.to(device=DEVICE);
