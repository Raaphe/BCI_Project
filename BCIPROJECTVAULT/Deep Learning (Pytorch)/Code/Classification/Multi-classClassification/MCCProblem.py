from helper_functions import plot_predictions, plot_decision_boundary;
import torch;
import matplotlib.pyplot as plt;
from sklearn.datasets import make_blobs;
from sklearn.model_selection import train_test_split;
from torch import nn;


# set hyperparameters for data creation

NUM_CLASSES = 4;
NUM_FEATURES = 2;
RANDOM_SEED = 42;

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES, 
                            cluster_std=1.5, 
                            centers=NUM_CLASSES, 
                            random_state=RANDOM_SEED);


# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float32);
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor);


# 3. Split data into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, 
                                                                        y_blob, 
                                                                        test_size=0.2, 
                                                                        random_state=RANDOM_SEED);


# 4. Plot data
plt.figure(figsize=(10,7));
plt.scatter(X_blob[:,0], X_blob[:,1], c=y_blob,  cmap='RdYlBu');
# plt.show();

DEVICE = "cuda" if torch.cuda.is_available() else "cpu";

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true.to(DEVICE), y_pred.to(DEVICE)).sum().item();
    acc = (correct/len(y_pred)) * 100;
    return acc;


class MultiClassClassificationModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units = 8):
        """Initializes multi-class classification model.
        Args: 
            input_features (int): Number of output features to the model
            output_features (int): Number of output features (number of output classes)
            hidden_units (int): Number of hidden units between layers, default 8

        Returns:

        Example:

        """
        super().__init__();
    
        self.linear__layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear__layer_stack(x);


model = MultiClassClassificationModel(input_features=2, output_features=4, hidden_units=8).to(DEVICE);

 

loss_fn = torch.nn.CrossEntropyLoss();

# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1);
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1);


### In order to evaluate and train and test our model, 
### we need to convert our model's outputs (logits), to prediction probabilities to prediction labels.
#
### Logits (raw output of the model) -> Pred Probs (use `torch.softmax()`) -> Prediction labels
### (Take the argmax of the prediction probabilities).

X_blob_train, X_blob_test = X_blob_train.to(DEVICE), X_blob_test.to(DEVICE);
y_blob_train, y_blob_test = y_blob_train.to(DEVICE), y_blob_test.to(DEVICE);

def train(modelToTrain: nn.Module):

    modelToTrain.train();

    y_logits = modelToTrain(X_blob_train);
    y_pred_probs = torch.softmax(y_logits, dim=1);
    y_preds = torch.argmax(y_pred_probs, dim=1);

    loss = loss_fn(y_logits, y_blob_train);
    acc = accuracy_fn(y_pred=y_preds, y_true=y_blob_train);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    return (loss, acc)


def test(modelToTrain: nn.Module):

    modelToTrain.eval();
    test_logits = modelToTrain(X_blob_test);
    y_test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1);

    loss = loss_fn(test_logits, y_blob_test);
    acc = accuracy_fn(y_true=y_blob_test, y_pred=y_test_preds);

    return (loss, acc)


torch.manual_seed(42);
torch.cuda.manual_seed(42);
epochs = 50;

for epoch in range(epochs):

    (train_loss, train_acc) = train(model);

    with torch.inference_mode():
        (test_loss, test_acc) = test(model);

    if (epoch % 10 == 0) :
        print(f"Epoch: {epoch} | train loss: {train_loss:4f} | train acc: {train_acc:.2f}% | test loss: {test_loss:4f} | test acc: {test_acc:.2f}% |")


with torch.inference_mode():

    model.eval();
    plt.figure(figsize=(12,6));
    plt.subplot(1,2,1);
    plt.title("Train");
    plot_decision_boundary(model, X_blob_train, y_blob_train);
    plt.subplot(1,2,2);
    plt.title("Test");
    plot_decision_boundary(model, X_blob_test, y_blob_test);
    plt.show();

