import matplotlib.pyplot as plt;
import torch;
import torchvision.datasets as datasets;
import torchvision;
from torch import nn;
from fashion_mnist_cnn_v1 import FashionMNISTModel2;
from pathlib import Path;

test_data = datasets.FashionMNIST(
    root="data",
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None,
);

MODEL_SAVE_PATH = "/Users/raphe/Library/Mobile Documents/com~apple~CloudDocs/BCI_Project/models/mnist_model.pth";

model:FashionMNISTModel2 = FashionMNISTModel2(input_shape=1,device="cpu", hidden_units=30, output_shape=10);
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=True));
model.eval();

with torch.inference_mode():
    labels = test_data.classes;

    torch.manual_seed(42);
    fig = plt.figure(figsize=(9,9));
    rows, cols = 4,4;
    for i in range(1, rows*cols+1):
        random_idx = torch.randint(0, len(test_data), size=[1]).item();

        label = test_data[random_idx][1];
        image = test_data[random_idx][0].unsqueeze(dim=1);

        # This makes a prediction, 
        # takes the output and turns it into a single int representing the predicted class, 
        # we then use the index to find the corresponding class name
        predicted_class = labels[model(image).argmax(dim=1)];

        fig.add_subplot(rows, cols, i);
        plt.imshow(image.squeeze(), cmap="gray");
        plt.title(predicted_class);
        plt.axis(False);    

    plt.show()



# Make preds
