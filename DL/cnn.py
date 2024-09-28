import numpy as np

# Activation Function
def relu(x):
    """
    Applies the Rectified Linear Unit (ReLU) activation function.
    
    Parameters:
    x (np.ndarray): Input array.
    
    Returns:
    np.ndarray: Output array where each element is the maximum between 0 and the input.
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Computes the derivative of the ReLU activation function.
    
    Parameters:
    x (np.ndarray): Input array.
    
    Returns:
    np.ndarray: Derivative array where each element is 1 if input > 0, else 0.
    """
    return (x > 0).astype(float)

# Convolutional Layer
class ConvLayer:
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        """
        Initializes the convolutional layer with a specified number of filters and filter size.
        
        Parameters:
        num_filters (int): Number of convolutional filters.
        filter_size (int): Size of each filter (assumed square).
        stride (int): Stride of the convolution.
        padding (int): Padding added to both sides of the input.
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        # Initialize filters with random values scaled by the filter size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / np.sqrt(filter_size * filter_size)
        self.biases = np.zeros(num_filters)
    
    def iterate_regions(self, image):
        """
        Generates all possible regions of the input image that the filters will convolve over.
        
        Parameters:
        image (np.ndarray): Input image of shape (height, width).
        
        Yields:
        tuple: A tuple containing the image region and its top-left corner coordinates (i, j).
        """
        h, w = image.shape
        h_padded, w_padded = h + 2 * self.padding, w + 2 * self.padding
        image_padded = np.pad(image, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        out_h = (h_padded - self.filter_size) // self.stride + 1
        out_w = (w_padded - self.filter_size) // self.stride + 1

        for i in range(0, out_h * self.stride, self.stride):
            for j in range(0, out_w * self.stride, self.stride):
                region = image_padded[i:i + self.filter_size, j:j + self.filter_size]
                yield region, i, j

    def forward(self, input):
        """
        Performs the forward pass of the convolutional layer.
        
        Parameters:
        input (np.ndarray): Input image of shape (height, width).
        
        Returns:
        np.ndarray: Output feature maps after convolution.
        """
        self.last_input = input
        h, w = input.shape
        h_padded, w_padded = h + 2 * self.padding, w + 2 * self.padding
        image_padded = np.pad(input, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        out_h = (h_padded - self.filter_size) // self.stride + 1
        out_w = (w_padded - self.filter_size) // self.stride + 1

        # Initialize output
        output = np.zeros((out_h, out_w, self.num_filters))

        # Vectorized convolution
        for region, i, j in self.iterate_regions(input):
            # Convolve the region with all filters
            output[i // self.stride, j // self.stride] = np.sum(region * self.filters, axis=(1, 2)) + self.biases

        self.last_output = output
        return output

    def backward(self, d_L_d_out, learning_rate):
        """
        Performs the backward pass of the convolutional layer.
        
        Parameters:
        d_L_d_out (np.ndarray): Gradient of the loss with respect to the output.
        learning_rate (float): Learning rate for parameter updates.
        
        Returns:
        np.ndarray: Gradient of the loss with respect to the input.
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_biases = np.zeros_like(self.biases)
        d_L_d_input = np.zeros_like(self.last_input)

        h, w = self.last_input.shape
        h_padded, w_padded = h + 2 * self.padding, w + 2 * self.padding
        image_padded = np.pad(self.last_input, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        d_L_d_input_padded = np.pad(d_L_d_input, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for region, i, j in self.iterate_regions(self.last_input):
            # Calculate gradients
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i // self.stride, j // self.stride, f] * region
                d_L_d_biases[f] += d_L_d_out[i // self.stride, j // self.stride, f]
                d_L_d_input_padded[i:i + self.filter_size, j:j + self.filter_size] += d_L_d_out[i // self.stride, j // self.stride, f] * self.filters[f]

        # Remove padding from gradient w.r. to input
        if self.padding != 0:
            d_L_d_input = d_L_d_input_padded[self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_L_d_input = d_L_d_input_padded

        # Update filters and biases
        self.filters -= learning_rate * d_L_d_filters
        self.biases -= learning_rate * d_L_d_biases

        return d_L_d_input

# Fully Connected Layer
class FullyConnected:
    def __init__(self, input_size, output_size):
        """
        Initializes the fully connected layer with random weights and zero biases.
        
        Parameters:
        input_size (int): Number of input neurons.
        output_size (int): Number of output neurons.
        """
        # Initialize weights with He initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros(output_size)
    
    def forward(self, input):
        """
        Performs the forward pass of the fully connected layer.
        
        Parameters:
        input (np.ndarray): Input data, can be multi-dimensional.
        
        Returns:
        np.ndarray: Output data after linear transformation.
        """
        self.last_input = input
        self.last_input_flat = input.flatten()
        return np.dot(self.last_input_flat, self.weights) + self.biases

    def backward(self, d_L_d_out, learning_rate):
        """
        Performs the backward pass of the fully connected layer.
        
        Parameters:
        d_L_d_out (np.ndarray): Gradient of the loss with respect to the output.
        learning_rate (float): Learning rate for parameter updates.
        
        Returns:
        np.ndarray: Gradient of the loss with respect to the input.
        """
        # Gradient w.r. to biases
        d_L_d_biases = d_L_d_out
        # Gradient w.r. to weights
        d_L_d_weights = np.outer(self.last_input_flat, d_L_d_out)
        # Gradient w.r. to input
        d_L_d_input = np.dot(self.weights, d_L_d_out).reshape(self.last_input.shape)

        # Update weights and biases
        self.weights -= learning_rate * d_L_d_weights
        self.biases -= learning_rate * d_L_d_biases

        return d_L_d_input

# Convolutional Neural Network
class CNN:
    def __init__(self, input_shape, num_classes):
        """
        Initializes the CNN with a convolutional layer and a fully connected layer.
        
        Parameters:
        input_shape (tuple): Shape of the input data (height, width).
        num_classes (int): Number of output classes.
        """
        self.conv = ConvLayer(num_filters=8, filter_size=3, stride=1, padding=1)
        # Calculate the size after convolution (same padding, stride 1)
        conv_output_size = input_shape[0] * input_shape[1] * 8
        self.fc = FullyConnected(input_size=conv_output_size, output_size=num_classes)
    
    def forward(self, input):
        """
        Performs the forward pass through the CNN.
        
        Parameters:
        input (np.ndarray): Input image of shape (height, width).
        
        Returns:
        np.ndarray: Output predictions from the network.
        """
        # Apply convolutional layer
        conv_out = self.conv.forward(input)
        # Apply ReLU activation
        activated = relu(conv_out)
        self.activated = activated  # Store for backward pass
        # Flatten the activated output
        flattened = activated.flatten()
        # Apply fully connected layer
        output = self.fc.forward(flattened)
        return output
    
    def train(self, input, target, learning_rate=0.01):
        """
        Trains the CNN on a single input-target pair using backpropagation.
        
        Parameters:
        input (np.ndarray): Input image.
        target (np.ndarray): Target output (one-hot encoded).
        learning_rate (float): Learning rate for parameter updates.
        
        Returns:
        float: The loss value for this training step.
        """
        # Forward pass
        output = self.forward(input)
        # Compute Mean Squared Error (MSE) loss
        loss = np.mean((output - target) ** 2)

        # Compute gradient of loss w.r. to output
        d_L_d_out = 2 * (output - target) / output.size  # Normalize by output size

        # Backward pass through fully connected layer
        d_L_d_fc_input = self.fc.backward(d_L_d_out, learning_rate)

        # Reshape gradient to match activated layer
        d_L_d_activated = d_L_d_fc_input.reshape(self.activated.shape)
        # Backprop through ReLU
        d_L_d_conv = d_L_d_activated * relu_derivative(self.conv.last_output)
        # Backward pass through convolutional layer
        d_L_d_input = self.conv.backward(d_L_d_conv, learning_rate)

        return loss

# usage
if __name__ == "__main__":
    input_shape = (28, 28)  
    num_classes = 10     

    # intialize the model
    model = CNN(input_shape, num_classes)

    # Generate random input and target data for demonstration
    np.random.seed(42)  # For reproducibility
    input_data = np.random.rand(*input_shape)
    target_label = np.random.randint(0, num_classes)
    target_data = np.zeros(num_classes)
    target_data[target_label] = 1  # One-hot encoding
    learning_rate = 0.01

    # Train the model on the sample data
    loss = model.train(input_data, target_data, learning_rate)
    print(f"Training loss: {loss:.4f}")
