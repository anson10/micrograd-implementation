# micrograd-implementation

This repository contains my personal implementation of **micrograd**, inspired by Andrej Karpathy’s tiny autograd engine.  
The goal of this project is to deeply understand **how neural networks actually compute gradients**, how **backpropagation works at the scalar level**, and how a full **MLP (Multi-Layer Perceptron)** can be built entirely from scratch with no external deep-learning libraries.

---

# What is micrograd?

**micrograd** is a *minimal automatic differentiation engine* that operates on **scalar values**, builds a **computational graph**, and performs **reverse-mode autodiff** (backpropagation).

It is intentionally simple:

- no tensors  
- no vectorization  
- no CUDA  
- no PyTorch / TensorFlow  
- just pure Python, basic math, and recursion

Despite being tiny, it contains everything needed to train neural networks.

---

# Math Behind micrograd

## **Computational Graph**

Every operation creates a new `Value` node:

```

z = x * y + w

```

forms a graph like:

```

x ─┐
├─ (*) ──┐
y ─┘        ├─ (+) → z
w ──────────┘

````

Each node stores:

- `data` → the number  
- `grad` → the derivative of the final output w.r.t this node  
- `_prev` → parents (for backprop)  
- `_backward()` → function that applies the local gradient rule  

---

##  **Automatic Differentiation (Backpropagation)**

Backprop is done by:

1. Performing a forward pass and storing the graph  
2. Doing a **topological sort**  
3. Calling `_backward()` on each node in reverse order  

Example:

\[
z = x \cdot y
\]

Then:

\[
\frac{\partial z}{\partial x} = y,\quad \frac{\partial z}{\partial y} = x
\]

micrograd implements exactly these rules manually.

---

##  **Activation Function: tanh**

A neuron uses:

\[
\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
\]

Derivative:

\[
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)
\]

This is coded in:

```python
def tanh(self):
    t = ...
    self.grad += (1 - t**2) * out.grad
````

---

## **Gradient Descent**

After computing gradients:

[
w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}
]

Your MLP updates every weight manually using `.grad`.

---

#  Code Structure

### **Value Class**

The core engine:

* Stores data + gradient
* Supports addition, multiplication, subtraction, power, tanh, exp
* Builds a graph of operations
* Implements backprop

### **Neuron**

A single neuron:

* Has a list of `Value` weights
* Computes weighted sum + bias
* Applies `tanh` activation

### **Layer**

A collection of neurons:

* Maps inputs → outputs
* Returns either a list or a single Value

### **MLP (Multi-Layer Perceptron)**

Stacks layers to form a neural network:

* input dimension → hidden layers → output layer
* Uses only the `Value` operations

---

# Purpose of This Repo

This project is meant for:

* Learning how **backpropagation really works**
* Understanding how deep learning libraries compute gradients
* Experimenting with a complete MLP engine written from scratch
* Visualizing training on simple datasets like `make_moons`

The goal is **education**, not performance.

---

# ▶️ Example: Training on `make_moons`

A demo script is included where:

* The dataset is generated via scikit-learn
* Labels are converted to ±1
* The MLP is trained using manual gradient descent
* Decision boundaries can be plotted
---


Just tell me!
```
