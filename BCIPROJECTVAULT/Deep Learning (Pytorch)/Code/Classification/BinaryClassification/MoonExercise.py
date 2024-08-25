from torchmetrics import Accuracy;
import matplotlib.pyplot as plt;
from torch import nn;
import pandas as pd;
import torch

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true.to(DEVICE), y_pred.to(DEVICE)).sum().item();
    acc = (correct/len(y_pred)) * 100;
    return acc;

# Setup device agnostic code
DEVICE = "cuda" if torch.cuda.is_available() else "cpu";

# Setup random seed
RANDOM_SEED = 42



# Create a dataset with Scikit-Learn's make_moons()
from sklearn.datasets import make_moons
     
x_input, y_target = make_moons(
    n_samples=1000, 
    random_state=RANDOM_SEED, 
    noise=0.07);

# Turn data into a DataFrame
data_df = pd.DataFrame({
    "X0": x_input[:,0], 
    "X1": x_input[:,1], 
    "y":y_target});

data_df.head();

# Visualize the data on a scatter plot
plt.scatter(x_input[:,0], x_input[:,1], c=y_target, cmap='RdYlBu')

# Turn data into tensors of dtype float
x_input = torch.from_numpy(x_input).type(torch.float32).to(DEVICE);
y_target = torch.from_numpy(y_target).type(torch.float32).to(DEVICE);

# Split the data into train and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split

     

x_train, x_test, y_train, y_test = train_test_split(
    x_input, 
    y_target, 
    random_state=RANDOM_SEED,
    test_size=0.2);

x_train = x_train.to(DEVICE);
x_test = x_test.to(DEVICE);
y_train = y_train.to(DEVICE);
y_test = y_test.to(DEVICE);

# Inherit from nn.Module to make a model capable of fitting the moon data
class MoonModelV0(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Linear(out_features=12,in_features=2),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=12),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=1),
        );
    

    def forward(self, x):
        return self.layers(x);
        

# Instantiate the model
## Your code here ##

model = MoonModelV0().to(DEVICE);

# Setup loss function
loss_fn = nn.BCEWithLogitsLoss();
acc_fn = Accuracy(task="multiclass", num_classes=2).to(DEVICE);

# Setup optimizer to optimize model's parameters
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1);

def train(modelToTrain: MoonModelV0) -> tuple:
    modelToTrain.train();
    logits = modelToTrain(x_train.to(DEVICE)).squeeze();
    y_pred = torch.round(torch.sigmoid(logits));
    loss = loss_fn(logits, y_train);
    acc = acc_fn(y_pred, y_train.int());
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    return (loss, acc)

def test(modelToTrain:MoonModelV0) -> tuple:
    modelToTrain.eval();
    logits = modelToTrain(x_test.to(DEVICE)).squeeze();
    y_pred = torch.round(torch.sigmoid(logits));
    loss = loss_fn(logits, y_test);
    acc = acc_fn(y_pred, y_test.int());
    return (loss, acc)



epochs = 1000;

for epoch in range(epochs):

    (train_loss, train_acc) = train(model);
    (test_loss, test_acc) = test(model);

    if (epoch % 100 == 0):
        print(f"Epoch : {epoch} | train loss: {train_loss:4f} | train acc: {train_acc:.2f}% | test loss : {test_loss:4f} | test acc : {test_acc:.2f}% |");



# Plot the model predictions
import numpy as np

def plot_decision_boundary(model, X, y):
  
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Source - https://madewithml.com/courses/foundations/neural-networks/ 
    # (with modifications)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), 
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # mutli-class
    else: 
        y_pred = torch.round(torch.sigmoid(y_logits)) # binary
    
    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, x_test, y_test)
plt.show()