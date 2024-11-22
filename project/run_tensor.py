"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

from turtle import forward
import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)
class Network(minitorch.Module):
    """A simple feedforward neural network with two hidden layers and ReLU activations."""

    def __init__(self, hidden_size):
        """Initializes the network's layers."""
        super().__init__()
        self.layer1 = Linear(2, hidden_size)
        self.layer2 = Linear(hidden_size, hidden_size)
        self.layer3 = Linear(hidden_size, 1)

    def forward(self, x):
        """Defines the forward pass through the network."""
        middle1 = self.layer1(x).relu()
        middle2 = self.layer2(middle1).relu()
        return self.layer3(middle2).sigmoid()


class Linear(minitorch.Module):
    """A linear (fully connected) layer for neural networks."""

    def __init__(self, in_size, out_size):
        """Initializes"""
        super().__init__()
        self.bias = RParam(out_size)
        self.weights = RParam(in_size, out_size)
        self.out_size = out_size

    def forward(self, x):
        """Performs the forward pass through the linear layer."""
        return (self.weights.value.view(1, *x.shape[1:], self.out_size) * x.view(*x.shape, 1)).sum(1).view(x.shape[0], self.out_size) + self.bias.value





def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)