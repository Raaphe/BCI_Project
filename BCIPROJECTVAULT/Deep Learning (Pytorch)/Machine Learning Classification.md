
---

## Architecture of a Classification Network
---


| **Hyperparameter**                      | **Binary Classification**                                                                                                                                                                                                                                       | **Multiclass classification**                                                                                                                             |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input layer shape** (`in_features`)   | Same as number of features (e.g. 5 for age, sex, height, weight, smoking status in heart disease prediction)                                                                                                                                                    | Same as binary classification                                                                                                                             |
| **Hidden layer(s)**                     | Problem specific, minimum = 1, maximum = unlimited                                                                                                                                                                                                              | Same as binary classification                                                                                                                             |
| **Neurons per hidden layer**            | Problem specific, generally 10 to 512                                                                                                                                                                                                                           | Same as binary classification                                                                                                                             |
| **Output layer shape** (`out_features`) | 1 (one class or the other)                                                                                                                                                                                                                                      | 1 per class (e.g. 3 for food, person or dog photo)                                                                                                        |
| **Hidden layer activation**             | Usually [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU) (rectified linear unit) but [can be many others](https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions)                                    | Same as binary classification                                                                                                                             |
| **Output activation**                   | [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) ([`torch.sigmoid`](https://pytorch.org/docs/stable/generated/torch.sigmoid.html) in PyTorch)                                                                                                          | [Softmax](https://en.wikipedia.org/wiki/Softmax_function) ([`torch.softmax`](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) in PyTorch) |
| **Loss function**                       | [Binary crossentropy](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression) ([`torch.nn.BCELoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) in PyTorch)                                       | Cross entropy ([`torch.nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) in PyTorch)                        |
| **Optimizer**                           | [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) (stochastic gradient descent), [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) (see [`torch.optim`](https://pytorch.org/docs/stable/optim.html) for more options) | Same as binary classification                                                                                                                             |



![[Pasted image 20240804100811.png]]![[Pasted image 20240804101329.png]]![[Pasted image 20240804102100.png]]

### Section on Logits
---
Our model outputs are going to be raw **Logits**.

We can convert these **Logits** into **prediction probabilities** by passing them to some kind of activation function (e.g. sigmoid for binary classification and softmax for multiclass classification).

Then, we can convert our model's prediction probabilities to **Prediction labels** by either rounding them, or taking the `argmax()`.

## Model is guessing
---
From the metrics our model is not learning. So to inspect it we make some predictions and make them visual. To do so we're going to import a function called [`plot_decision_boundary()`](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py).

$$z = \sum_{i}(w_i \cdot x_i) + b$$


### Improving our Model 
---
These options all come from the model's perspective and not the data. And because these options are all values we (as machine learning engineers and data scientists) can change, they are referred to as **hyperparameters**.

- Add more layers - give our models more chances to learn patterns in the data
- Add more hidden units - go from 5 hidden units to 10 hidden units
- Fit for longer (more epochs)
- Changing the activation functions
- Change the learning rate (`lr` = the magnitude of the step)
- Change the loss function

![[Pasted image 20240810165805.png]]

Let's say in our case we have this model which is underperforming :

```python
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__();

        
        self.layer_1 = nn.Linear(in_features=2, out_features=5); 
        self.layer_2 = nn.Linear(in_features=5, out_features=1); 

    def forward(self, x):
        return self.layer_2(self.layer_1(x)); # x -> layer_1 -> layer_2 -> output

/// RESULTS IN

> | Epoch : 990 | Loss : 0.69298 , Acc : 51.12% | Test Loss : 0.69468 , Test Acc : 45.50% |
```

Let's try and improve this model by 
- Adding more hidden units : 5 -> 10
- Increase the number of layers : 2 -> 3
- Increase the number of epochs : 100 -> 1000

... Still bad ...


### The missing piece : Non-Linearity
---

"What patterns could you draw if you were given an infinite amount of straight and non-straight lines?"
Or in machine learning terms, an infinite (but really it is finite) amount of linear and non-linear functions?

### Building a model with non-linearity
---
- Linear = straight line
- Non-Linear = non-straight lines

Artificial neural networks are a large combinations of linear and non-linear functions which are potentially able to find patterns in data.

### Replicating non-linear activation functions
---

Neural Networks, rather than us telling the model what to learn, we give it the tools to discover patterns in data and it tries to figure out the patterns on its own.
And these tools are linear and non-linear functions.

### A few more classification metrics (to evaluate our classification model)
---


- Accuracy - out of 100 samples how many does our model get right?
- Precision 
- Recall 
- F1-Score 
- Confusion Matrix
- Classification report

**key: tp** = True Positive, **tn** = True Negative, **fp** = False Positive, **fn** = False Negative

| Metric Name      | Metric Formula                                                      | Code                                                              | When to use                                                                                                                   |
| ---------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Accuracy         | $$\text{Accuracy} = \frac{tp + tn}{tp + tn + fp + fn}$$             | `torchmetrics.Accuracy()` or `sklearn.metrics.accuracy_score()`   | Default metric for **classification problems**. Not the best for imbalanced classes.                                          |
| Precision        | $$\text{Precision} = \frac{tp}{tp + fp}$$                           | `torchmetrics.Precision()` or `sklearn.metrics.precision_score()` | Higher precision leads to less false positive                                                                                 |
| Recall           | $$\text{Recall} = \frac{tp}{tp + fn}$$                              | `torchmetrics.Recall()` or `sklearn.metrics.recall_score()`       | Higher recall leads to less false negatives                                                                                   |
| F1-Score         | $$\text{F1-Score} = \frac{Precision * Recall}{Precision + Recall}$$ | `torchmetrics.F1Score()` or `sklearn.metrics.f1_score()`          | Combination of precision recall, usually a good overall metric for a classification model                                     |
| Confusion Matrix | NA                                                                  | `torchmetrics.ConfusionMatrix()`                                  | When comparing predictions to truth labels to see where model gets confused. Can be hard to use with large number of classes. |


#### Example of `torchmetrics` 

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchmetrics import Accuracy;

# Setup metric with device agnostic code.
torchmetric_accuracy = Accuracy().to(DEVICE);

# Calculate accuracy 
torchmetric_accuracy(y_preds, y_blob_test);
```