# Deep Dream

This is Deep Dream! A fun project to visualize how a neural network dreams. The idea is to take a trained CNN (Inception in this case) and till a certain hidden layer calculate the gradients of that layer’s activation function and backpropagate, apply Gradient Ascent to maximize the activation function of that layer and update the image by adding those gradients to it.

Sample output

![Image of Yaktocat](https://github.com/aashishksahu/Deep-Learning/blob/master/Deep%20Dream/download.jpg?raw=true)
