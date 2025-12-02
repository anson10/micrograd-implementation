import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from micrograd_engine import Value, MLP

# --- Generate dataset ---
X, y_int = make_moons(n_samples=200, noise=0.1)
y = np.where(y_int == 0, -1.0, 1.0)  # map to {-1, 1}

# --- Create model ---
model = MLP(2, [16, 16, 1])   # 2→16→16→1 network
print("Total parameters:", len(model.parameters()))

# Convert X to Value objects
def to_value_row(x):
    return [Value(v) for v in x]

X_value = [to_value_row(row) for row in X]

# --- Training loop ---
lr = 0.05
epochs = 2000
history = []

for epoch in range(epochs):

    # Forward pass
    ypred = [model(xv) for xv in X_value]

    # Compute MSE loss
    loss = Value(0.0)
    for yt, yp in zip(y, ypred):
        out = yp if not isinstance(yp, list) else yp[0]
        loss += (out - Value(yt))**2

    # Backprop
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # Gradient descent
    for p in model.parameters():
        p.data -= lr * p.grad

    # Logging
    history.append(loss.data)
    if epoch % 200 == 0:
        print(f"Epoch {epoch}  Loss = {loss.data:.4f}")

# --- Plot loss ---
plt.plot(history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()

# --- Decision boundary plot ---
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_value = [[Value(a), Value(b)] for a, b in grid]

Z = np.array([(model(v)[0] if isinstance(model(v), list) else model(v)).data
              for v in grid_value])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
plt.title("Decision Boundary")
plt.show()
