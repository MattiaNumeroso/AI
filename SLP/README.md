# Single Layer Perceptron from scratch

Hi! 👋

This is a very basic implementation of a **Single Layer Perceptron (SLP)** from scratch. The goal of this example is to approximate a simple linear function:  

f(x) = 2*x + 2


## What’s inside

- All the steps of **forward propagation** and **backpropagation** are explicitly written out, so you can see exactly how the neuron learns.
- We train the neuron using a simple dataset based on the target function and watch how it adjusts its weights and bias.

## Visualizations

There are **two plots** included in the code:  

1. **Training Error per Epoch** – This shows how the error decreases as the perceptron learns.
2. **Predictions vs Target Function** – This compares the output of our neuron with the actual target function. Ideally, the two curves overlap perfectly, showing that our SLP learned the function.

## Note

Keep in mind that a **single layer perceptron can only approximate linear functions**. It won’t be able to learn non-linear functions, no matter how long you train it.

---

Feel free to play around with the code and experiment with different weights, learning rates, or datasets!
