## NOT OPTIMIZED CODE - FOR EDUCATIONAL PURPOSES ONLY
import math
import random
import time
from typing import List, Tuple, Callable

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

# --- 2. CORE MATHEMATICAL OPERATIONS ---

class MatrixMath:
    """Handles raw mathematical operations using explicit loops for educational complexity."""
    
    @staticmethod
    def dot_product(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Performs matrix multiplication O(N^3) using nested loops."""
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        
        if cols_A != rows_B:
            raise ValueError("Incompatible matrices for multiplication.")
            
        # Initialize result matrix with zeros
        result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
        
        # Complex nested loop structure
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    @staticmethod
    def transpose(A: List[List[float]]) -> List[List[float]]:
        """Transposes a matrix."""
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    @staticmethod
    def apply_activation(matrix: List[List[float]], func: Callable) -> List[List[float]]:
        """Applies a function element-wise across a matrix."""
        return [[func(val) for val in row] for row in matrix]

# --- 3. NEURAL NETWORK ARCHITECTURE ---

class NeuralNetwork:
    """A multi-layer perceptron built completely from scratch."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with random values between -1 and 1
        self.W1 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.W2 = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        
        # Caching for backpropagation
        self.hidden_layer_output = []
        self.final_output = []

    # Activation functions
    def sigmoid(self, x: float) -> float:
        # Clamping to prevent math overflow
        x = max(min(x, 500), -500) 
        return 1.0 / (1.0 + math.exp(-x))

    def sigmoid_derivative(self, x: float) -> float:
        return x * (1.0 - x)

    def forward_pass(self, X: List[List[float]]) -> List[List[float]]:
        """Pushes data through the network layers."""
        # Input to Hidden Layer
        hidden_input = MatrixMath.dot_product(X, self.W1)
        self.hidden_layer_output = MatrixMath.apply_activation(hidden_input, self.sigmoid)
        
        # Hidden to Output Layer
        final_input = MatrixMath.dot_product(self.hidden_layer_output, self.W2)
        self.final_output = MatrixMath.apply_activation(final_input, self.sigmoid)
        
        return self.final_output

    @timing_decorator
    def train(self, X: List[List[float]], y: List[List[float]], epochs: int, learning_rate: float):
        """The core training loop including Forward and Backward Propagation."""
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # 1. Forward Pass
            output = self.forward_pass(X)
            
            # 2. Calculate Error (Mean Squared Error components)
            output_error = []
            for i in range(len(y)):
                row_error = []
                for j in range(len(y[0])):
                    row_error.append(y[i][j] - output[i][j])
                output_error.append(row_error)
            
            # 3. Backpropagation (Output Layer)
            output_delta = []
            for i in range(len(output)):
                row_delta = []
                for j in range(len(output[0])):
                    derivative = self.sigmoid_derivative(output[i][j])
                    row_delta.append(output_error[i][j] * derivative)
                output_delta.append(row_delta)
                
            # 4. Backpropagation (Hidden Layer)
            W2_transposed = MatrixMath.transpose(self.W2)
            hidden_error = MatrixMath.dot_product(output_delta, W2_transposed)
            
            hidden_delta = []
            for i in range(len(self.hidden_layer_output)):
                row_delta = []
                for j in range(len(self.hidden_layer_output[0])):
                    derivative = self.sigmoid_derivative(self.hidden_layer_output[i][j])
                    row_delta.append(hidden_error[i][j] * derivative)
                hidden_delta.append(row_delta)
                
            # 5. Weight Updates (Gradient Descent)
            # Update W2
            hidden_transposed = MatrixMath.transpose(self.hidden_layer_output)
            W2_adjustment = MatrixMath.dot_product(hidden_transposed, output_delta)
            for i in range(len(self.W2)):
                for j in range(len(self.W2[0])):
                    self.W2[i][j] += W2_adjustment[i][j] * learning_rate
                    
            # Update W1
            X_transposed = MatrixMath.transpose(X)
            W1_adjustment = MatrixMath.dot_product(X_transposed, hidden_delta)
            for i in range(len(self.W1)):
                for j in range(len(self.W1[0])):
                    self.W1[i][j] += W1_adjustment[i][j] * learning_rate

            if epoch % 500 == 0:
                # Calculate basic loss for logging
                loss = sum([sum([abs(val) for val in row]) for row in output_error]) / len(y)
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

# --- 4. EXECUTION SCRIPT ---

if __name__ == "__main__":
    print("Initializing Data Generator and Neural Network...")
    
    # Generate XOR Problem Dataset
    # X = Inputs, y = Expected Outputs
    training_data_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    training_data_y = [[0], [1], [1], [0]]
    
    # Initialize Network: 2 inputs, 4 hidden nodes, 1 output
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    # Train the network
    nn.train(training_data_X, training_data_y, epochs=2000, learning_rate=0.5)
    
    # Test the network post-training
    print("\n--- Final Predictions ---")
    predictions = nn.forward_pass(training_data_X)
    for i in range(len(training_data_X)):
        print(f"Input: {training_data_X[i]} | Target: {training_data_y[i][0]} | Prediction: {predictions[i][0]:.4f}")