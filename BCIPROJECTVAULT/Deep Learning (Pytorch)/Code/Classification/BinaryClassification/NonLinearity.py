from helper_functions import plot_decision_boundary;
import matplotlib.pyplot as plt;
from sklearn.datasets import make_circles;
import torch;
from torch import nn;
from sklearn.model_selection import train_test_split;

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true.to(DEVICE), y_pred.to(DEVICE)).sum().item();
    acc = (correct/len(y_pred)) * 100;
    return acc;

DEVICE = "cuda" if torch.cuda.is_available() else "cpu";

n_samples = 1000;

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42);

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu');

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);


X_train = X_train.to(DEVICE);
X_test = X_test.to(DEVICE);
y_train = y_train.to(DEVICE);
y_test = y_test.to(DEVICE);

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__();

        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # ReLU is a non linear activation function


    def forward(self, x):
        # Where should you put the non-linear activation function?
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))));


model = CircleModel().to(DEVICE);

loss_fn = nn.BCEWithLogitsLoss();

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.90);

epochs = 1000;

for epoch in range(epochs):
    model.train()
    # Forward pass
    logits = model(X_train).squeeze();
    y_pred = torch.round(torch.sigmoid(logits));
    loss = loss_fn(logits, y_train);
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    optimizer.zero_grad();
    loss.backward()
    optimizer.step();

    model.eval()
    with torch.inference_mode():

        logits_test = model(X_test).squeeze();
        test_preds = torch.round(torch.sigmoid(logits_test));
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)

        if (epoch % 50 == 0):
            print(f"Epoch {epoch} |loss : {loss_fn(logits_test, y_test)} | Train Accuracy : {acc:.2f}% | Test Accuracy : {test_acc:.2f}% ")


            


plt.figure(figsize=(12,6));
plt.subplot(1,2,1);
plt.title("Train");
plot_decision_boundary(model, X_train, y_train);
plt.subplot(1,2,2);
plt.title("Test");
plot_decision_boundary(model, X_test, y_test)


plt.show();

# if you get back to this and youre lost I was basically just trying to run the model myself Im at 12;12;49 but what he
# Might show for the next video or 2 you might of already figured out fr