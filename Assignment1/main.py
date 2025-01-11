import numpy as np

weights = None
bias = None

def initialize_weights(num_features):
    """Initialize weights to zeros and bias to 0."""
    global weights, bias
    weights = np.zeros(num_features)
    bias = 0

def predict(X):
    """Predict the target variable y for the given input X."""
    global weights, bias
    if weights is None or bias is None:
        raise ValueError("Model is not initialized. Please call 'initialize_weights' first.")
    return np.dot(X, weights) + bias

def calculate_loss(X, y):
    """Compute Mean Squared Error (MSE) loss."""
    num_samples = len(y)
    predictions = predict(X)
    return np.mean((predictions - y)**2)

def calculate_gradients(X, y):
    """Compute the gradient of the cost function w.r.t. weights and bias."""
    global weights, bias
    num_samples = len(y)
    predictions = predict(X)
    grad_weights = (1 / num_samples) * np.dot(X.T, (predictions - y))
    grad_bias = (1 / num_samples) * np.sum(predictions - y)
    return grad_weights, grad_bias

def update_weights(X, y, lr):
    """Update weights and bias using gradient descent."""
    global weights, bias
    grad_weights, grad_bias = calculate_gradients(X, y)
    weights -= lr * grad_weights
    bias -= lr * grad_bias

def train(X, y, lr=0.001, epochs=1000):
    """Train the model by adjusting the weights and bias using gradient descent."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y should be numpy arrays.")
    if len(X.shape) != 2:
        raise ValueError("X should be a 2D array.")
    if len(y.shape) != 1:
        raise ValueError("y should be a 1D array.")
    
    num_samples, num_features = X.shape
    initialize_weights(num_features)
    
    # Gradient descent loop
    for epoch in range(epochs):
        update_weights(X, y, lr)
        if epoch % 100 == 0 or epoch == epochs - 1:
            loss = calculate_loss(X, y)
            print(f"Epoch {epoch+1}/{epochs}: Loss = {loss:.4f}")

def evaluate(X, y):
    """Evaluate the model by calculating the Mean Squared Error (MSE)."""
    predictions = predict(X)
    mse = np.mean((predictions - y) ** 2)
    return mse


# Training data
X = np.array([[1, 2, 4], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 11]])
y = np.array([3, 5, 7, 9, 11])

# Train the model
train(X, y, lr=0.01, epochs=1000)

# Display the learned parameters (weights and bias)
print("Weights:", weights)
print("Bias:", bias)

# Test the model with new data
test_X = np.array([[1, 2, 4]])
predictions = predict(test_X)
print("Predicted value:", predictions)

# Calculate train error
train_error = evaluate(X, y)
print(f"Training error (MSE): {train_error:.4f}")

# Test model error
test_error = predictions - y[0]
print(f"Train prediction error: {test_error:.4f}")
