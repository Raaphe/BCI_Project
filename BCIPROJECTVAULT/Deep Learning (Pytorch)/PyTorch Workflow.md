
---

This section will cover the broad workflow when building a model using PyTorch. There are a few steps that we will cover:


| Topic                                                        | Contents                                                                                                                                     |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Getting data Ready**                                    | Data can be almost anything but to get started we're going to create a simple straight line                                                  |
| **2. Building a model**                                      | Here we'll create a model to learn patterns in the data, we'll also choose a **loss function**, **optimizer** and build a **training loop**. |
| **3. Fitting the model to data (training)**                  | We've got data and a model, now let's let the model (try to) find patterns in the (**training**) data.                                       |
| **4. Making predictions and evaluating a model (inference)** | Our model's found patterns in the data, let's compare its findings to the actual (**testing**) data.                                         |
| **5. Saving and loading a model**                            | You may want to use your model elsewhere, or come back to it later, here we'll cover that.                                                   |
| **6. Putting it all together**                               | Let's take all of the above and combine it.                                                                                                  |
![[Pasted image 20240804011707.png]]

## 1. Data (preparing and loading)
---

Data can be almost anything... in machine learning.

* Excel spreadsheet
* Images of any kind
* Videos 
* Audio like songs or podcasts
* DNA
* Text

>Machine learning is a game of two parts 
>   1. Get data into a numerical representation
>   2. Build a model to learn patterns in that numerical representation.


To showcase this, let's create some *known* data using the linear regression formula. We'll use a linear regression formula to make a straight line with known **Parameters**.


```python
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

// OUTPUT

tensor([[0.3000],
        [0.3140],
        [0.3280],
        [0.3420],
        [0.3560],
        [0.3700],
        [0.3840],
        [0.3980],
        [0.4120],
        [0.4260]])
torch.Size([50, 1])

tensor([[0.0000],
        [0.0200],
        [0.0400],
        [0.0600],
        [0.0800],
        [0.1000],
        [0.1200],
        [0.1400],
        [0.1600],
        [0.1800]])
torch.Size([50, 1])
```

### Three Datasets (important)
---

 One of most important steps in a machine learning project is creating a training and test set (and when required, a validation set).

Each split of the dataset serves a specific purpose:

| Split              | Purpose                                                                                                                     | Amount of total data | How often is it used? |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- | -------------------- | --------------------- |
| **Training set**   | The model learns from this data (like the course materials you study during the semester)                                   | ~60-80%              | Always                |
| **Validation set** | The model gets tuned on this data (like the practice exam you take before the final exam)                                   | ~10-20%              | Often but not always  |
| **Testing set**    | The model gets evaluated on this data to test what it has learned (like the final exam you take at the end of the semester) | ~10-20%              | Always                |
 ![[Pasted image 20240804013237.png]]

```python
### Splitting data into training and tests sets (one of the most important concepts in machine learning in general)


# Create a train/test split
train_split = int(0.8 * len(X));
X_train, y_train = X[:train_split], y[:train_split];
x_test, y_test = X[train_split:], y[train_split:];
  

print(f"Len of X_train {len(X_train)}")
print(f"Len of y_train {len(X_train)}")
print(f"Len of X_test {len(x_test)}")
print(f"Len of y_test {len(y_test)}")

// OUTPUT

Len of X_train 40
Len of y_train 40
Len of X_test 10
Len of y_test 10
```

#### Visualizing data
---


How might we better visualize our data? This is where the data explorer motto comes in (iykyk).

```python
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
    plt.legend(prop={"size": 14});
    
    plt.show();

  
  
plot_predictions();
```

## Build a model
---

What our model does :

- Start with random numbers ( weight and bias )
- Look at training data and adjust the random values to better represent (or get close to) the ideal values (the weight and bias values we used to create the data)

[How does it do so](# How Does Gradient Descent and Backpropagation Work Together?) ;
1. Gradient descent 
2. Backpropagation

![[Pasted image 20240804032155.png]]

### PyTorch model building essentials
---

- `torch.nn` - Contains all of the building blocks for computational graphs (a neural network can be considered a computational graph)
- `torch.nn.Parameter` - what parameters should our model try and learn (often a PyTorch layer from `torch.nn` will set this for us)
- `torch.nn.Module` - The base class for all neural network module, if you subclass it, you should overwrite `forward()`.
- `torch.optim` - this is where optimizers in PyTorch live, they will help with gradient descent
- `def forward()` All nn.Module subclasses require you to overwrite `forward`, this method defines what happens in the forward computation.

### [Cheat sheet](https://pytorch.org/tutorials/beginner/ptcheat.html)
---

![[Pasted image 20240804033144.png]]

## Making predictions
---

To check our model's predictive power, let's see how well it predicts `y_test` based on `x_test`. When we pass data through our model, it's going to run it through the forward method.

## Training Models
---

The whole idea of training is for a model to move from some *unknown* parameters (these may be random) to some *known* parameters.

Or in other words from a poor representation of the data to a better representation of the data. One way to measure how poor or how wrong your model is, is loss functions.

> [!NOTE] 
> Loss functions may also be called cost function or criterion in different areas. for our case, we're going to refer to it as a loss function. 
> 
> **Loss Function:** A function to measure how wrong your model's predictions are to the ideal outputs, lower is better.

Things we need to train:

- **Loss Function**: Defined above
- **Optimizer**: Takes into account the loss of a model and adjusts the model's parameters (e.g. weight and bias) in our case to improve the loss function.
	- inside the optimizer you'll often have to set two parameters:
		- `params` - the model parameters you'd like to optimize, for example `param=model_0.parameters()`
		- `lr` (learning rate) - the learning rate is a hyperparameter that defines how big/small the optimizer changes the parameters with each step (a small `lr` results in small changes, a large `lr` results in large changes)

And specifically in PyTorch, we need:
- A training loop
- A testing loop


> **Q:** Which loss function and optimizer should I use?
> **A:** This will be problem specific. But with experience, you'll get an idea of what works and what doesn't with your particular problem set.
> 
> For example, for a regression problem (like ours), a loss function of `nn.L1Loss()` and an optimizer like `torch.optim.SGD()` will suffice.
> 
> But for a classification problem like classifying whether a photo is of a dog or a cat, you'll likely want to use a loss function of `nn.BCELoss()` (binary cross entropy loss).


### Building a testing/training loop
---

A couple of things we need in a training loop:
0. Loop through the data
1. Forward pass (this involves data moving through our model's `forward()` function)  - also called forward propagation
2. Calculate the loss (compare forward pass predictions to ground truth labels)
3. Optimizer zero grad
4. Loss backward - move backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss (**backpropagation**).
5. Optimizer step - use the optimizer to adjust our model's parameters to try and improve the loss (**gradient descent**).

![[Pasted image 20240804045800.png]]

![[Pasted image 20240804074447.png]]

## Saving and loading Models
---

There are three main methods you should know about for saving and loading models in PyTorch.

1. `torch.save()` - allows you to save a PyTorch object in Python's pickle format
2. `torch.load()` - allows you to load a saved PyTorch object.
3. `torch.nn.Module.load_state_dict()` - this allows to load a model's saved state dictionnary


##### Saving
```python
# Saving our Pytorch model

# 1. Create models directory
MODEL_PATH = Path("models");
MODEL_PATH.mkdir(oarents=True, exist_ok=True);

# 2 . Create model save path
MODEL_NAME = "model_01_PyTorch_workflow.pth";
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME;

# 3. Save the model state dict
print(f"saving model to {MODEL_SAVE_PATH}");
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH);

// OUTPUT

> saving model to models\model_01_PyTorch_workflow.pth
```

##### Loading a PyTorch Model
---

Since we saved our model's `state_dict()` rather than the entire model, we'll create a new instance of our model class and load the saved `state_dict()` into that.

```python
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

// OUTPUT

tensor([[True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True]])

```


   