import numpy as np
import time
from typing import Callable

# --- 1. UTILITY AND DECORATOR FUNCTIONS ---

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure the execution time of complex loops."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[METRIC] {func.__name__} executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

# --- 2. NEURAL NETWORK ARCHITECTURE ---

class NeuralNetwork:
    """A multi-layer perceptron optimized with NumPy."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with random values between -1 and 1
        self.W1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.W2 = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        
        # Caching for backpropagation
        self.hidden_layer_output = None
        self.final_output = None

    # Activation functions
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # np.clip prevents math overflow warnings
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        # x is the output of the sigmoid function, not the input
        return x * (1.0 - x)

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Pushes data through the network layers using vectorization."""
        # Input to Hidden Layer
        hidden_input = np.dot(X, self.W1)
        self.hidden_layer_output = self.sigmoid(hidden_input)
        
        # Hidden to Output Layer
        final_input = np.dot(self.hidden_layer_output, self.W2)
        self.final_output = self.sigmoid(final_input)
        
        return self.final_output

    @timing_decorator
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float):
        """The core training loop with vectorized Forward and Backward Propagation."""
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # 1. Forward Pass
            output = self.forward_pass(X)
            
            # 2. Calculate Error
            output_error = y - output
            
            # 3. Backpropagation (Output Layer)
            output_delta = output_error * self.sigmoid_derivative(output)
                
            # 4. Backpropagation (Hidden Layer)
            hidden_error = np.dot(output_delta, self.W2.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)
                
            # 5. Weight Updates (Gradient Descent)
            # Update W2
            self.W2 += np.dot(self.hidden_layer_output.T, output_delta) * learning_rate
                    
            # Update W1
            self.W1 += np.dot(X.T, hidden_delta) * learning_rate

            if epoch % 500 == 0:
                # Calculate basic loss for logging
                loss = np.mean(np.abs(output_error))
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

# --- 3. EXECUTION SCRIPT ---

if __name__ == "__main__":
    print("Initializing Data Generator and Neural Network...")
    
    # Generate XOR Problem Dataset
    # X = Inputs, y = Expected Outputs (Converted to NumPy arrays)
    training_data_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_data_y = np.array([[0], [1], [1], [0]])
    
    # Initialize Network: 2 inputs, 4 hidden nodes, 1 output
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    # Train the network
    nn.train(training_data_X, training_data_y, epochs=2000, learning_rate=0.5)
    
    # Test the network post-training
    print("\n--- Final Predictions ---")
    predictions = nn.forward_pass(training_data_X)
    for i in range(len(training_data_X)):
        print(f"Input: {training_data_X[i].tolist()} | Target: {training_data_y[i][0]} | Prediction: {predictions[i][0]:.4f}")