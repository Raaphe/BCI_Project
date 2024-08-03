
----

```python
What is Deep Learning?
```

**Machine learning** is turning things (data) into numbers and **finding patterns** in those numbers. **Deep Learning** is a subset of **machine learning**.

![[Pasted image 20240722193737.png]]

```python
Why use Machine Learning (or Deep Learning)?
```

- Good reason: Why not? 
- Better Reason : For a complex problem, can you think of all the rules that would come into play if you were to write a traditional program such as driving, recognizing images, formulating sentences, etc.

> Google's #1 rule for machines learning : "If you can build a **simple rule-based** system that doesn't require machine learning, do that." 
> - A wise software engineer.

What is machine learning good for then? ;

- **Problems with a long list of rules** -- When the traditional approach fails, machine learning/deep learning may help.
- **Continually changing environments** -- Deep learning can adapt ('learn') to new scenarios
- **Discovering insights within the large collections of data** -- Can you imagine trying to hand-craft rules for what 101 different kinds of food looks like?

What deep learning is not good for? 

- **When you need explainability** -- The patterns learned by a deep learning model are typically uninterpreted by a human.
- **When the traditional approach is a better option** -- if you can accomplish what you need with a simple rule-based system.
- **When errors are unacceptable** -- Since the outputs of deep learning models aren't always predictable.
- **When you don't have much data** -- deep learning models usually require a fairly large amount of data to produce great results.

## Machine learning vs. Deep learning 
---
![[Pasted image 20240722195314.png]]
![[Pasted image 20240722195351.png]]

## What are neural networks ?
---

When you have input of data, before it is used in a neural network it needs to be turned into numbers. Once this is done, we pass it through our neural networks that is composed of layers (input layer, hidden layers, output layers). Our network will "learn" patterns in our input and then it will output those representation outputs. We can convert these outputs into human friendly/ human legible data. 

![[Pasted image 20240722195928.png]]

### Anatomy of Neural Networks
---

![[Pasted image 20240722200214.png]]


## Types of Learning
---
![[Pasted image 20240722200342.png]]

We will learn Supervised learning as Well as Transfer learning within this course. Also another model that isn't mentioned here is [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning).
### Supervised learning
---

This is when you have data and labels where your model needs to have lots of examples of what your output should look like. ex: You have 1000 images of cats and dogs that you know which are which and you pass to your model to discern.

### Unsupervised Learning & Self-Supervised Learning
---

Is when you just have the data with no labels. In this case your model learns inherent and fundamental patterns in data without label. Learning solely on the data itself.

### Transfer Learning
---

Taking the patterns that one model has learned from a dataset and passing it along to another model. Very powerful apparently.


## Some Use-Cases
---

![[Pasted image 20240722201646.png]]

