import numpy as np

np.random.seed(42)
class0 = np.random.multivariate_normal([-1, -1], np.eye(2), 10)
class1 = np.random.multivariate_normal([1, 1], np.eye(2), 10)
X = np.vstack((class0, class1))
y = np.array([0]*10 + [1]*10).reshape(-1, 1)

def manual_train_test_split(X, y, test_size=0.5):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = manual_train_test_split(X, y)

def initialize_parameters():
    return {
        'W1': np.random.randn(2, 1),
        'b1': np.random.randn(1, 1),
        'W2': np.random.randn(1, 1),
        'b2': np.random.randn(1, 1)
    }

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X, params):
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, params['W2']) + params['b2']
    return {'Z1': Z1, 'A1': A1, 'Z2': Z2}

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def backward(X, y, params, cache):
    m = X.shape[0]
    A1, Z2 = cache['A1'], cache['Z2']
    
    dZ2 = -2*(y - Z2)/m
    dW2 = np.dot(A1.T, dZ2)
    db2 =np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = np.dot(dZ2, params['W2'].T)
    dZ1 =dA1 * A1 * (1 - A1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0)
    
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

def update_params(params, grads, lr):
    for key in params:
        params[key] -= lr * grads[f'd{key}']
    return params

# 4. Training Loop
def train(X, y, epochs=1000, lr=0.1):
    params = initialize_parameters()
    for epoch in range(epochs):
        cache = forward(X, params)
        loss = compute_loss(y, cache['Z2'])
        grads = backward(X, y, params, cache)
        params = update_params(params, grads, lr)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    return params

params = train(X_train, y_train)

# 5. Evaluation
def evaluate(X, y, params):
    cache = forward(X, params)
    loss = compute_loss(y, cache['Z2'])
    print(f"\nTest MSE: {loss:.6f}")
    
    print("\nPredictions vs True:")
    for i in range(len(y)):
        print(f"Sample {i}: True={y[i][0]}, Pred={cache['Z2'][i][0]:.4f}")

evaluate(X_test, y_test, params)