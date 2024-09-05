# Import tqdm for progress bar
from tqdm.auto import tqdm
import torch;
from torch import nn;
from fashion_mnist_cnn_v1 import FashionMNISTModel2;
from torchvision import datasets;
import torchvision;
from torch.utils.data import DataLoader;
from torchmetrics import ConfusionMatrix;
from mlxtend.plotting import plot_confusion_matrix;
import matplotlib.pyplot as plt;

test_data = datasets.FashionMNIST(
    root="data",
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None,
);

DEVICE = "cuda" if torch.cuda.is_available() else "cpu";
MODEL_SAVE_PATH = "/Users/raphe/Library/Mobile Documents/com~apple~CloudDocs/BCI_Project/models/mnist_model.pth";
BATCH_SIZE = 32;

test_dataloader = DataLoader(
    batch_size= BATCH_SIZE,
    shuffle= False, # no need to shuffle when evaluating
    dataset=test_data
);


model: FashionMNISTModel2 = FashionMNISTModel2(input_shape=1, output_shape=10, hidden_units=30);
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=True));
model.to(DEVICE);
model.eval();

with torch.inference_mode():
    labels = test_data.classes;

# 1. Make predictions with trained model
y_preds = []
model.eval()

with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc="Making predictions"):
    # Send data and targets to target device
    X, y = X.to(DEVICE), y.to(DEVICE)
    # Do the forward pass
    y_logit = model(X)
    # Turn predictions from logits -> prediction probabilities -> predictions labels
    
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)

    # Put predictions on CPU for evaluation
    y_preds.append(y_pred.cpu())

print(f"y_preds {y_preds}");
# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)

print(f"y_pred_tensor: {y_pred_tensor}");

conf_matrix = ConfusionMatrix(num_classes=len(labels), task="multiclass");
conf_mat_tensor = conf_matrix(preds=y_pred_tensor, target=test_data.targets);

fig, ax = plot_confusion_matrix(
    conf_mat=conf_mat_tensor.numpy(), # matplotlib operates on numpy arrays
    class_names=labels,
    figsize=(10,7)
);

plt.show();