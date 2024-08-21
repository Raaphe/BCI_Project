import requests;
from pathlib import Path;
import pandas as pd;
import torch;
from torch import nn;
import matplotlib.pyplot as plt;
import sklearn
from sklearn.datasets import make_circles;
from sklearn.model_selection import train_test_split;

## Make classification data and get it ready

# Make 1000 samples
n_samples = 1000;

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42);

circles = pd.DataFrame({"X1": X[:,0], 
                        "X2" : X[:,1],
                        "label": y});

print(circles.head(10))

# Visualize

plt.scatter(x=X[:,0],
            y=X[:,1],
            c="r");
# plt.show();

# View the first example of features and labels
X_sample = X[0];
y_sample = y[0];

print(f"Values for one sample of X: {X_sample} and same for y: {y_sample}");
print(f"Shapes for one sample of X: {X_sample.shape}");

X = torch.from_numpy(X).type(torch.float32);
y = torch.from_numpy(y).type(torch.float);

print("\n");
print(X[:5]);
print(y[:5]);

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 0.2 = 20% of data will be test & 80% will be train
                                                    random_state=42
                                                    ); 


# Building a model

# Lets build a model to classify the blue and red dots

DEVICE = "cuda" if torch.cuda.is_available() else "cpu";

X_train.to(DEVICE);
X_test.to(DEVICE);
y_test.to(DEVICE);
y_train.to(DEVICE);

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__();
        # Create 2 nn.Linear layers capable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5); # Takes in 2 features and upscales to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1); # Output layer


    # Define a forward method that outlines the forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x)); # x -> layer_1 -> layer_2 -> output

model = CircleModel().to(DEVICE);

# Let's replicate the model above using nn.Sequential()
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(DEVICE);

# Make some predictions
model.eval();
with torch.inference_mode():
    untrained_preds = model(X_test.to(DEVICE));

print("\n");
print(f"Lenght of the predictions {len(untrained_preds)}, Shape : {untrained_preds.shape}");
print(f"Lenght of test samples: {len(X_test)}, Shape : {X_test.shape}");

print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}\nFirst 10 Labels:\n{y_test[:10]}");

# Finding Loss function and optimizer
# Which loss function or optimizer should I use for a classification model?
# Again... this is problem specific
# For example for regression I might use MAE 
# For Classification,  you might wanna try Binary Cross Entropy or Categorical Cross Entropy (cross entropy)
# As a reminder, the loss function measures how wrong our model's predictions are.

# And for optimizers,  two of the most common and useful are SGD and Adam, however, PyTorch has many built-in options.

# * For the loss function we're going to use `torch.nn.BECWIthLogitsLoss()`, for more info on what binary cross entropy  (BCE) is check out this article - https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
# * For a definition on what a logit is in deep learning - https://stackoverflow.com/a/52111173
# * For different optimizers see `torch.optim`

# loss_fn = nn.BCELoss(); # BCELoss = requires inputs to have gone through the sigmoid activation function once prior to input to BCELoss
loss_fn = nn.BCEWithLogitsLoss(); # BCEWithLogitsLoss = sigmoid activation function built into loss function.

optimizer = torch.optim.SGD(params=model.parameters(), 
                            lr=0.1);

# Calculate accuracy - FORMULA : Accuracy = True positive / (True Positive + True Negative) * 100
# Def : Out of 100 examples how many predictions does our model get right.

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true.to(DEVICE), y_pred.to(DEVICE)).sum().item();
    acc = (correct/len(y_pred)) * 100;
    return acc;

def train(model_p):
    model_p.train();

    # forward
    y_logits = model_p(X_train.to(DEVICE)).squeeze();
    y_preds = torch.round(torch.sigmoid(y_logits)); # turn logits -> pred probs -> pred labels

    ## For our prediction probability values, we need to perform a range-style rounding on them:
    # `y_pred_probss` >= 0.5, y = 1 (class 1)
    # `y_pred_probss` < 0.5, y = 0 (class 0)

    # Loss/accuracy
    loss = loss_fn(y_logits, 
                   y_train.to(DEVICE)); # Our loss function is `BCEWithLogitsLoss()` so it expects the logits, not predictions

    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_preds);

    # Optimizer Zero Grad
    optimizer.zero_grad();

    # loss backward (backpropagation)
    loss.backward()

    # Optimizer step (gradient descent)
    optimizer.step();
    return (loss, acc);

 

def test(model_p):
    model_p.eval()
    with torch.inference_mode():
        # 1. forward pass
        test_logits = model_p(X_test.to(DEVICE)).squeeze();
        test_pred = torch.round(torch.sigmoid(test_logits));

        # Calculate the test loss/acc
        test_loss = loss_fn(test_logits.to(DEVICE), y_test.to(DEVICE));
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred);
        return (test_loss, test_acc);

torch.manual_seed(42);
torch.cuda.manual_seed(42);

epochs = 100;

for epoch in range(epochs):
    (loss, acc) = train(model);
    (test_loss, test_acc) = test(model);

    if (epoch % 10 == 0):
        print(f"| Epoch : {epoch} | Loss : {loss:.5f} , Acc : {acc:.2f}% | Test Loss : {test_loss:.5f} , Test Acc : {test_acc:.2f}% |");


if Path("helper_function.py").is_file():
    print("the code already exists, skipping..");
else:
    print("Downloading helper functions");
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py");
    with open("helper_functions.py", "wb") as f:
        f.write(request.content);

from helper_functions import plot_predictions, plot_decision_boundary;

plt.figure(figsize=(12,6));
plt.subplot(1,2,1);
plt.title("Train");
plot_decision_boundary(model, X_train, y_train);
plt.subplot(1,2,2);
plt.title("Test");
plot_decision_boundary(model, X_test, y_test)
# plt.show();


class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__();

        self.layer1 = nn.Linear(in_features=2, out_features=10);
        self.layer2 = nn.Linear(in_features=10, out_features=10);
        self.layer3 = nn.Linear(in_features=10, out_features=1);

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x))); # this way of writting operations leverages speed ups where possible BTS
         


model_1 = CircleModelV1().to(DEVICE);

loss_fn = nn.BCEWithLogitsLoss();
optimizer = torch.optim.SGD(params=model_1.parameters(), 
                            lr=0.1);

torch.manual_seed(42);
torch.cuda.manual_seed(42);

X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE);
X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE);


for epoch in range(1000):

    # Train
    model_1.train()

    # 1. Forward Pass
    logits =  model_1(X_train).squeeze();
    y_pred = torch.round(torch.sigmoid(logits));

    # 2. Calculate the loss/acc
    loss = loss_fn(logits, y_train); # Loss function works with logits
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred);
    
    # 3. Optimizer zero grad
    optimizer.zero_grad();

    
    loss.backward();

    optimizer.step();

    model_1.eval()

    with torch.inference_mode():
        logitsTest = model_1(X_test).squeeze();
        test_pred = torch.round(torch.sigmoid(logitsTest));
    
        test_loss = loss_fn(logitsTest, y_test);
        test_acc = accuracy_fn(y_pred=test_pred, y_true=y_test);


    if (epoch % 20 == 0):
        print(f"| Epoch : {epoch} | Loss : {loss:.5f} , Acc : {acc:.2f}% | Test Loss : {test_loss:.5f} , Test Acc : {test_acc:.2f}% |");



plt.figure(figsize=(12,6));
plt.subplot(1,2,1);
plt.title("Train");
plot_decision_boundary(model_1, X_train, y_train);
plt.subplot(1,2,2);
plt.title("Test");
plot_decision_boundary(model_1, X_test, y_test)
plt.show()


