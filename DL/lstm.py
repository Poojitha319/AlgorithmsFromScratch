import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the LSTM network with specified input, hidden, and output sizes.
        
        Parameters:
        - input_size (int): Size of the input vector (e.g., vocabulary size).
        - hidden_size (int): Number of units in the hidden layer.
        - output_size (int): Size of the output vector (e.g., vocabulary size).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Xavier Initialization for weights to maintain the variance of activations
        limit_f = np.sqrt(1 / (input_size + hidden_size))
        self.Wf = np.random.uniform(-limit_f, limit_f, (hidden_size, input_size + hidden_size))  # Forget gate weights
        self.Wi = np.random.uniform(-limit_f, limit_f, (hidden_size, input_size + hidden_size))  # Input gate weights
        self.Wc = np.random.uniform(-limit_f, limit_f, (hidden_size, input_size + hidden_size))  # Cell candidate weights
        self.Wo = np.random.uniform(-limit_f, limit_f, (hidden_size, input_size + hidden_size))  # Output gate weights

        # Biases initialized to zero
        self.bf = np.zeros((hidden_size, 1))  # Forget gate bias
        self.bi = np.zeros((hidden_size, 1))  # Input gate bias
        self.bc = np.zeros((hidden_size, 1))  # Cell candidate bias
        self.bo = np.zeros((hidden_size, 1))  # Output gate bias

        # Output layer weights and biases
        limit_y = np.sqrt(1 / hidden_size)
        self.Wy = np.random.uniform(-limit_y, limit_y, (output_size, hidden_size))  # Output weights
        self.by = np.zeros((output_size, 1))  # Output bias

    @staticmethod
    def sigmoid(x):
        """
        Computes the sigmoid activation function in a numerically stable way.
        
        Parameters:
        - x (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Output after applying sigmoid.
        """
        # To prevent overflow, clip the input values
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Computes the softmax function in a numerically stable way.
        
        Parameters:
        - x (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Output probabilities after applying softmax.
        """
        # Subtract the max for numerical stability
        shiftx = x - np.max(x, axis=0, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def forward(self, inputs, h_prev, c_prev):
        """
        Performs the forward pass of the LSTM over a sequence of inputs.
        
        Parameters:
        - inputs (list of np.ndarray): List of input vectors for each time step.
        - h_prev (np.ndarray): Previous hidden state.
        - c_prev (np.ndarray): Previous cell state.
        
        Returns:
        - dict: A dictionary containing all intermediate variables needed for backpropagation.
        """
        # Dictionaries to store intermediate values
        xs, hs, cs, zs, fs, is_, c_tildes, os, ys = {}, {}, {}, {}, {}, {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)

        for t, x in enumerate(inputs):
            xs[t] = x
            # Concatenate previous hidden state and current input
            zs[t] = np.vstack((hs[t - 1], xs[t]))

            # Forget gate
            fs[t] = self.sigmoid(np.dot(self.Wf, zs[t]) + self.bf)
            # Input gate
            is_[t] = self.sigmoid(np.dot(self.Wi, zs[t]) + self.bi)
            # Cell candidate
            c_tildes[t] = np.tanh(np.dot(self.Wc, zs[t]) + self.bc)
            # Cell state
            cs[t] = fs[t] * cs[t - 1] + is_[t] * c_tildes[t]
            # Output gate
            os[t] = self.sigmoid(np.dot(self.Wo, zs[t]) + self.bo)
            # Hidden state
            hs[t] = os[t] * np.tanh(cs[t])
            # Output (unnormalized)
            ys[t] = np.dot(self.Wy, hs[t]) + self.by

        return {'xs': xs, 'hs': hs, 'cs': cs, 'zs': zs, 'fs': fs,
                'is_': is_, 'c_tildes': c_tildes, 'os': os, 'ys': ys}

    def backward(self, cache, targets, learning_rate=1e-2):
        """
        Performs the backward pass (Backpropagation Through Time) of the LSTM.
        
        Parameters:
        - cache (dict): Dictionary containing all intermediate variables from the forward pass.
        - targets (list of np.ndarray): List of target vectors for each time step.
        - learning_rate (float): Learning rate for parameter updates.
        
        Returns:
        - float: The total loss for the sequence.
        """
        # Unpack cache
        xs, hs, cs, zs, fs, is_, c_tildes, os, ys = (cache['xs'], cache['hs'], cache['cs'],
                                                    cache['zs'], cache['fs'], cache['is_'],
                                                    cache['c_tildes'], cache['os'], cache['ys'])

        # Initialize gradients with zeros
        dWf, dWi, dWc, dWo = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wc), np.zeros_like(self.Wo)
        dbf, dbi, dbc, dbo = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bc), np.zeros_like(self.bo)
        dWy, dby = np.zeros_like(self.Wy), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])
        dc_next = np.zeros_like(cs[0])
        loss = 0

        # Iterate backward through time
        for t in reversed(range(len(xs))):
            # Compute loss (Cross-Entropy Loss with softmax)
            y = ys[t]
            p = self.softmax(y)
            loss += -np.log(p[targets[t], 0] + 1e-8)  # Add epsilon for numerical stability

            # Gradient of loss w.r.t y
            dy = np.copy(p)
            dy[targets[t]] -= 1  # derivative of loss w.r.t y

            # Gradients for output layer
            dWy += np.dot(dy, hs[t].T)
            dby += dy

            # Gradient w.r.t hidden state
            dh = np.dot(self.Wy.T, dy) + dh_next

            # Gradient w.r.t cell state
            dc = dh * os[t] * (1 - np.tanh(cs[t])**2) + dc_next

            # Gradients for output gate
            do = dh * np.tanh(cs[t])
            do_raw = do * os[t] * (1 - os[t])

            # Gradients for cell state
            dc_tilde = dc * is_[t]
            dc_tilde_raw = dc_tilde * (1 - c_tildes[t]**2)

            # Gradients for input gate
            di = dc * c_tildes[t]
            di_raw = di * is_[t] * (1 - is_[t])

            # Gradients for forget gate
            df = dc * cs[t - 1]
            df_raw = df * fs[t] * (1 - fs[t])

            # Accumulate gradients for gates
            dWf += np.dot(df_raw, zs[t].T)
            dWi += np.dot(di_raw, zs[t].T)
            dWc += np.dot(dc_tilde_raw, zs[t].T)
            dWo += np.dot(do_raw, zs[t].T)

            dbf += df_raw
            dbi += di_raw
            dbc += dc_tilde_raw
            dbo += do_raw

            # Gradients w.r.t concatenated inputs
            dz = (np.dot(self.Wf.T, df_raw) +
                  np.dot(self.Wi.T, di_raw) +
                  np.dot(self.Wc.T, dc_tilde_raw) +
                  np.dot(self.Wo.T, do_raw))

            # Split gradients into h_prev and x
            dh_next = dz[:self.hidden_size, :]
            dc_next = dc * fs[t]

        # Clip gradients to prevent exploding gradients
        for dparam in [dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dWy, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update parameters using gradient descent
        self.Wf -= learning_rate * dWf
        self.Wi -= learning_rate * dWi
        self.Wc -= learning_rate * dWc
        self.Wo -= learning_rate * dWo
        self.bf -= learning_rate * dbf
        self.bi -= learning_rate * dbi
        self.bc -= learning_rate * dbc
        self.bo -= learning_rate * dbo
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby

        return loss

    def sample(self, seed, n):
        """
        Generates a sequence of indices by sampling from the LSTM's predictions.
        
        Parameters:
        - seed (int): The index to start the sequence with.
        - n (int): The length of the sequence to generate.
        
        Returns:
        - list of int: The generated sequence of indices.
        """
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        x = np.zeros((self.input_size, 1))
        x[seed] = 1
        indices = []

        for t in range(n):
            # Concatenate hidden state and input
            z = np.vstack((h, x))

            # Gates and cell computations
            f = self.sigmoid(np.dot(self.Wf, z) + self.bf)
            i = self.sigmoid(np.dot(self.Wi, z) + self.bi)
            c_tilde = np.tanh(np.dot(self.Wc, z) + self.bc)
            c = f * c + i * c_tilde
            o = self.sigmoid(np.dot(self.Wo, z) + self.bo)
            h = o * np.tanh(c)

            # Output computation
            y = np.dot(self.Wy, h) + self.by
            p = self.softmax(y)

            # Sampling the next index
            idx = np.random.choice(range(self.output_size), p=p.ravel())
            x = np.zeros((self.input_size, 1))
            x[idx] = 1
            indices.append(idx)

        return indices

# Example usage
if __name__ == "__main__":
    # Define network dimensions
    input_size = 100   # e.g., vocabulary size
    hidden_size = 128  # Number of LSTM units
    output_size = 100  # e.g., vocabulary size

    # Initialize the LSTM model
    lstm = LSTM(input_size, hidden_size, output_size)

    # Example forward pass
    sequence_length = 10
    inputs = [np.random.randn(input_size, 1) for _ in range(sequence_length)]  # Random input sequence
    h_prev = np.zeros((hidden_size, 1))  # Initial hidden state
    c_prev = np.zeros((hidden_size, 1))  # Initial cell state
    cache = lstm.forward(inputs, h_prev, c_prev)

    # Example backward pass
    targets = [np.random.randint(0, output_size) for _ in range(sequence_length)]  # Random target indices
    loss = lstm.backward(cache, targets, learning_rate=1e-2)
    print(f"Training loss: {loss:.4f}")

    # Example sampling
    seed = np.random.randint(0, input_size)  # Random seed index
    sampled_indices = lstm.sample(seed, 100)  # Generate a sequence of 100 indices
    print(f"Sampled indices: {sampled_indices}")
