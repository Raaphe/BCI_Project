---
aliases:
---
---


### Examples Computer Vision Problem
---


- Binary Classification - one thing or the other
- Multi-Class Classification - More than one thing or an other
- Object Detection - "Where's the thing we're looking for?"
- Image Segmentation - "What are the different sections on this image?"

### What is a CNN (Convolutional Neural Network)
---

**What is a CNN?**

> https://youtu.be/QzY57FaENXg?si=6_1m2ueSrWdIF8yw


Architecture of a CNN:

| Hyperparameters/Layer Type              | What does it do?                                                          | Typical Values                                                                                                                                                                                                             |
| --------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Input Image(s)                          | Target images you'd like to discover patterns in                          | Whatever you can take a photo (or video) of                                                                                                                                                                                |
| Input Layer                             | Takes in target images and preprocesses them for further layers.          | ```input_shape = [batch_size, image_width, image_heightm, color_channels]```               {channels last} or  `input_shape = [batch_size, image_width, image_heightm, color_channels]`                                    |
| Convolutional Layer                     | Extracts/learns the most important features from the target images        | <br><br>```<br>input_shape = [batch_size, image_width, image_height, color_channels]<br>```<br><br>(channels last) or <br><br>```<br>input_shape = [batch_size, color_channels, image_width, image_height];<br>```<br><br> |
| Hidden activation/non-linear activation | Adds non-linearity to learned features (non straight lines)               | Multiple, can create with `torch.nn.ConvXd()` (X can be multiple values)                                                                                                                                                   |
| Pooling Layer                           | Reduces the dimensionality of learned image features.                     | Usually Relu from `torch.nn.ReLU()`, though can be many more                                                                                                                                                               |
| Output Layer/Linear Layer               | Takes learned features and outputs them in shape of target labels/target. | `torch.nn.Linear(out_features=[number_of_classes])` (e.g. 3 for pizza, steak and sushi)                                                                                                                                    |
| Output Activation                       | Converts output logits to prediction probabilities                        | `torch.sigmoid()` (binary classification) or `torch.softmax()` (multi-class classification)                                                                                                                                |

