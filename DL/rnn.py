import numpy as np

class RNN:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Parameters initialization
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias

    def forward(self, inputs, h_prev):
        """
        Forward pass of the RNN
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)

        for t, x in enumerate(inputs):
            xs[t] = x
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)  # Hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by  # Unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # Probabilities for next chars

        return xs, hs, ps

    def backward(self, xs, hs, ps, targets):
        """
        Backward pass through time (BPTT)
        """
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])

        for t in reversed(range(len(xs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext  
            dhraw = (1 - hs[t] * hs[t]) * dh  
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  

        return dWxh, dWhh, dWhy, dbh, dby

    def sample(self, seed, n):
        """
        Sample a sequence of integers from the model
        """
        h = np.zeros((self.hidden_size, 1))
        x = np.zeros((self.input_size, 1))
        x[seed] = 1
        indices = []

        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            idx = np.random.choice(range(self.input_size), p=p.ravel())
            x = np.zeros((self.input_size, 1))
            x[idx] = 1
            indices.append(idx)

        return indices

input_size = 100  
output_size = 100  
hidden_size = 64  

rnn = RNN(input_size, output_size, hidden_size)


inputs = [np.random.randn(input_size, 1) for _ in range(10)]  
h_prev = np.zeros((hidden_size, 1))
xs, hs, ps = rnn.forward(inputs, h_prev)


targets = [np.random.randint(0, output_size) for _ in range(10)]  
dWxh, dWhh, dWhy, dbh, dby = rnn.backward(xs, hs, ps, targets)

# Example sampling
seed = np.random.randint(0, input_size)  
sampled_indices = rnn.sample(seed, 100)  
