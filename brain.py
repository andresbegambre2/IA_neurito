"""Red 3 -> oculta (ReLU) -> 3 (Softmax), sin interfaz gráfica."""
import numpy as np


def policy(data):
    return np.argmax(np.column_stack((data[:, 0], 1 - data[:, 1], data[:, 2])), axis=1)


def relu(values):
    return np.maximum(0, values)


def softmax(values):
    exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp_values / exp_values.sum(axis=1, keepdims=True)


class Brain:
    def __init__(self, hidden=6, seed=42):
        if hidden < 1:
            raise ValueError("Se requiere al menos una neurona oculta")
        self.rng = np.random.RandomState(seed)
        self.X = self.rng.rand(3000, 3)
        self.targets = policy(self.X)
        self.W1 = self.rng.randn(3, hidden) * 0.5
        self.b1 = np.zeros((1, hidden))
        self.W2 = self.rng.randn(hidden, 3) * 0.5
        self.b2 = np.zeros((1, 3))

    def predict(self, data):
        return softmax(relu(data @ self.W1 + self.b1) @ self.W2 + self.b2)

    def loss(self):
        p = self.predict(self.X)
        return float(-np.mean(np.log(p[np.arange(len(self.X)), self.targets] + 1e-10)))

    def train(self, epochs=1200, learning_rate=0.1, report=None):
        if epochs < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Las épocas y la tasa deben ser positivas y finitas")
        for epoch in range(epochs):
            z1 = self.X @ self.W1 + self.b1
            hidden = relu(z1)
            p = softmax(hidden @ self.W2 + self.b2)
            if report is not None and epoch % 200 == 0:
                report(epoch, float(-np.mean(np.log(p[np.arange(len(p)), self.targets] + 1e-10))))
            delta2 = p.copy()
            delta2[np.arange(len(p)), self.targets] -= 1
            delta2 /= len(p)
            grad_W2 = hidden.T @ delta2
            grad_b2 = delta2.sum(axis=0, keepdims=True)
            delta1 = (delta2 @ self.W2.T) * (z1 > 0)
            self.W1 -= learning_rate * (self.X.T @ delta1)
            self.b1 -= learning_rate * delta1.sum(axis=0, keepdims=True)
            self.W2 -= learning_rate * grad_W2
            self.b2 -= learning_rate * grad_b2
        return self
