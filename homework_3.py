"""# Introduction to Artificial Neural Networks

Please read the introdcution of neuronal networks of the book *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, p. 299-316.

Why have neural networks, even though they were invented early on, only now caught on?

What is a percepton and a threshold logic unit (TLU)? Try to define a linear function and a step function of your choice, use some values of your choice and explain what might be the result of the percepton. (maybe using max. two TLU's)

What is a fully connected layer and a output layer? Why can we easily combine the equations of multiple instances into a fully connected layer?

What problem did Marvin Minsky and Seymour Paper highlight that perceptrons could not solve? What is a possible solution?

What is a deep neuronal network? What are hidden layers? What means feedforward neural network (FNN).

Try to explain how backpropagation works! (In Addition, you can have a look to the following example, which tries manually to compute the backprogation of a simple linear network. https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ OR you can also read through the Google Colab [04_mnist_basics.ipynb](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb#scrollTo=t1DK6o-gckCy))

Why do we need activation functions, wouldn't it be easier just using linear functions?

## Ideas for the learning portfolio: 

1) For example, you could train a single TLU to classify iris flowers based on petal length and width in the !!!pyTorch!! environment.

2) You could add to our king county housepricing ML project a neuronal network and compare it to the other models.

1.	The recent rise in the popularity of neural networks is due to the increasing availability of large amounts of data, faster computing power, and improvements in algorithm development. These advancements have made neural networks more efficient and effective, which has led to their wider use in various applications.
2.	A perceptron is a type of neural network that consists of a single layer of threshold logic units (TLUs). A TLU is a neuron that computes a weighted sum of its inputs, adds a bias term, and applies a step function to the result. A step function is a binary function that outputs 1 or 0 based on whether its input is above or below a certain threshold. For example, a step function with a threshold of 0 would output 1 for any input greater than 0 and 0 for any input less than or equal to 0. We can use two TLUs with different weights and biases to implement a logical operation like AND, OR, or XOR.
3.	A fully connected layer is a layer in a neural network where each neuron is connected to every neuron in the previous layer. An output layer is the final layer of a neural network that produces the network's output (in keras I wouls call it the “dense layer”). We can easily combine the equations of multiple instances into a fully connected layer because each neuron in the layer uses the same input vector but with different weights and biases.
4.	Marvin Minsky and Seymour Papert pointed out that perceptrons could not solve non-linearly separable problems. One potential solution was to use multi-layer neural networks with non-linear activation functions, which could handle non-linearly separable problems. 
5.	A deep neural network is a neural network that has multiple hidden layers. Hidden layers perform intermediate computations between input and output layers. A feedforward neural network is a type of neural network where information flows from the input layer to the output layer without any feedback connections.
6.	Backpropagation is an algorithm that trains neural networks by computing the gradient of a loss function with respect to the network's weights and then using this gradient to update the weights to minimize the loss. The algorithm uses the chain rule of calculus to compute the gradient of the loss function with respect to each weight in the network.
7.	Activation functions are crucial in neural networks because they add non-linearity to the model. Without non-linearity, a neural network would be a linear combination of its inputs, which is not sufficient for modeling complex patterns in data. Activation functions allow neural networks to learn non-linear relationships between inputs and outputs, making them more effective for tasks like image recognition, natural language processing, and speech recognition.
"""

! [ -e /content ] && pip install -Uqq fastbook
import fastbook
from fastai.vision.all import *
from fastbook import *
fastbook.setup_book()
from sklearn import datasets
iris = datasets.load_iris()

"""# A traditional approach: training a digit classifier and learning pyTorch tensors.

For this assignment, I ask you to read the Google Colab [04_mnist_basics.ipynb](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb#scrollTo=t1DK6o-gckCy) to the beginning of the chapter *Stochastic Gradient Descent (SGD)*. 

First, try to summarize what we know about pyTorch tensors by trying to predict whether we have a 1 or a 7 in the MNIST dataset using a traditional rule-based programming approach. Therefore use pyTorch tensors for the entire tasks and fulfill the following steps:

1) Randomly split the MNIST dataset (1 and 7) into a training dataset and a test dataset in a ratio of 80:20.

2) Instead of using an optimal 1 or 7 with the mean over the training dataset, try to calculate the sum of the distances to all instances in the training set for each instance in the test dataset. You can use the L2 norm. 

3) For each instance in the test set, decide if it is a 1 or 7 and calculate the precision.

Do we get a similar good result?

PyTorch Tensors are arrays or matrices with multiple dimensions, like Numpy arrays, which are designed to work efficiently with deep learning algorithms. They are the primary building block in PyTorch, representing input data, intermediate computations, and model parameters. Tensors can be created on the CPU or GPU and manipulated using various mathematical operations such as matrix multiplication, addition, and subtraction. PyTorch also provides a range of functions for manipulating tensors, like slicing, indexing, and concatenation. Tensors support automatic differentiation, allowing gradients to be computed automatically during backpropagation, making them useful for deep learning models. Additionally, PyTorch tensors are highly compatible with NumPy, simplifying the conversion between the two libraries. Overall, PyTorch tensors are a versatile and powerful data structure that form the foundation of many deep learning applications.
"""

path = untar_data(URLs.MNIST)
Path.BASE_PATH = path
(path/'training').ls()
ones = (path/'training'/'1').ls().sorted()
sevens = (path/'training'/'7').ls().sorted()

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_image(path):
  img = Image.open(path)
  return np.array(img)

X_ones = np.array([load_image(p)for p in ones])

X_sevens = np.array([load_image(p) for p in sevens])

y_ones = np.full(len(X_ones), 1)
y_sevens = np.full(len(X_sevens), 7)

X = np.concatenate([X_ones, X_sevens])
y = np.concatenate([y_ones, y_sevens])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

print(f"Train set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

import torch

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

print(f"Train set type: {type(X_train)}")
print(f"Test set type: {type(X_test)}")

X_train, X_test = X_train.float(), X_test.float()
y_train, y_test = y_train.float(), y_test.float()
X_train /= 255
X_test /= 255
y_train /= 255
y_test /= 255

for name, data in [("Train set", X_train), ("Test set", X_test), ("Train labels", y_train), ("Test labels", y_test)]:
    print(f"{name} min: {data.min()}")
    print(f"{name} max: {data.max()}")

def mnist_distance(a, b):
    return ((a - b) ** 2).sum((-1, -2))

distances = []
for x in X_test:
    distances.append([mnist_distance(x, xt) for xt in X_train])


y_pred = []
for dist in distances:
    sum1 = sum([d for i, d in enumerate(dist) if y_train[i] == 1])
    sum7 = sum([d for i, d in enumerate(dist) if y_train[i] == 7])
    if sum1 < sum7:
        y_pred.append(1)
    else:
        y_pred.append(7)

precision = (y_pred == y_test).sum().item() / len(y_test)
print("Precision:", precision)

import torch 

def mnist_distance(a, b):
    return ((a - b) ** 2).sum((-1, -2))

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).clone().detach().to('cuda')
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).clone().detach().to('cuda')
y_train_tensor = torch.tensor(y_train, dtype=torch.long).clone().detach().to('cuda')

distances = mnist_distance(X_test_tensor[:, None], X_train_tensor)
sum1 = (distances[:, y_train_tensor == 1]).sum(axis=1)
sum7 = (distances[:, y_train_tensor == 7]).sum(axis=1)
y_pred = torch.where(sum1 < sum7, torch.tensor(1, dtype=torch.long).to('cuda'), torch.tensor(7, dtype=torch.long).to('cuda'))

precision = (y_pred == torch.tensor(y_test, dtype=torch.long).to('cuda')).sum().item() / len(y_test)
print("Precision:", precision)

"""# Stochastic Gradient Descent (SGD)

For this exercise I ask you to read the chapter Stochastic Gradient Descent (SGD) from the Google Colab 04_mnist_basics.ipynb in paralell. The chapter starts with a single TLU, compare p. 304 in "Hands on Machine Learning". Go through all 7 steps which are an easy example of how Stochastic Gradient Descent works.

Our goal is to train a single TLU, which can decide if one number is larger then the other one. Therefore we create 100 random pairs with pyTorch and create a target vector which is eather 1 or 0.

"""

x = torch.randn((100, 2))
y = torch.where(x[:,0] > x[:,1], 1.0, 0.0)
print(x)
print(y)

"""Your task is to create a function f that is a single TLU, meaning that it summarizes x with weights a, b, c:

$ax_0+bx_1+c$

In Addition we are using a *sigmoid()* function as step function.

$f = \text{sigmoid}(ax_0+bx_1+c)$
"""

import torch

def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))
    #calculate the sigmoid of x (z)

def f(x, params):
    a, b, c = params
    z = a*x[:,0] + b*x[:,1] + c
    return sigmoid(z)

"""In addition to our TLU function, we need a loss function. Your task is to implement a absolute difference loss function, $∑|x_i-y_i|$, which counts the number of wrong guesses."""

def mae(preds, targets):
    return torch.mean(torch.abs(preds - targets))

"""Try to train your single TLU with the absolute difference loss function, use the following code. Choose an appropriate step weight `lr` and try to explain what is happing in each line."""

lr = 1 #A learning rate of 1 means that on each iteration, the parameters (weights) of the model 
#will be updated by an amount equal to the gradient of the loss function multiplied by 1. 
#so the weights are basically adjusted by the value of the gradient of the loss function 
params = torch.randn(3).requires_grad_() #initialize random weights
#this code only works this way because it is a tensor object, which is a relatively special kind of object in python 
def apply_step(params, prn=True):
    preds = f(x, params)
    loss = mae(preds, y) 
    loss.backward() #calculates gradient of loss function in order to determine what change in weights allows 
    #for the fastest adjustment of weights to accurately find the pattern in the data. 
    #should the gradient of the loss function become lower, than we are probably finding better weights. 
    print(f"params before applying learning rate: {params.grad.data}")
    params.data -= lr * params.grad.data #updating the weights of the tensorflow object parans with the learning rate 
    print(f"params after applying learning rate: {params.data}")
    params.grad = None
    if prn: print(params);print(loss.item())
    return preds


for i in range(50): apply_step(params)

"""Write a line of code that counts the number of wrong predictions, rounding your predictions with *round()*."""

preds = f(x, params)
rounded_preds = torch.round(preds)
num_wrong = torch.sum(torch.abs(rounded_preds - y))

for i in range(len(y)):
    if rounded_preds[i] == y[i]:
        print(f"Sample {i}: Correct Prediction")
    else:
        print(f"Sample {i}: Incorrect Prediction")
