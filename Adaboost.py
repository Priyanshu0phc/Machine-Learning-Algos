import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt

def prepare_mnist_data():
    # Load MNIST data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
    X_train_0 = X[y == 0][:1000]
    X_train_1 = X[y == 1][:1000]
    y_train_0 = y[y == 0][:1000]
    y_train_1 = y[y == 1][:1000]
    
    X_test_0 = X[y == 0][1000:]
    X_test_1 = X[y == 1][1000:]
    y_test_0 = y[y == 0][1000:]
    y_test_1 = y[y == 1][1000:]
    
    X_train = np.vstack([X_train_0, X_train_1])
    y_train = np.hstack([y_train_0, y_train_1])
    X_test = np.vstack([X_test_0, X_test_1])
    y_test = np.hstack([y_test_0, y_test_1])
    
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    split_idx = int(0.8 * len(X_train))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]

    pca = PCA(n_components=5, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    return X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.direction = None
        self.beta = None
    
    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        if self.direction == 'left':
            predictions[X[:, self.feature_idx] > self.threshold] = -1
        else:
            predictions[X[:, self.feature_idx] <= self.threshold] = -1
            
        return predictions
    
    def find_best_split(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        best_error = float('inf')
        y_ada = np.where(y == 1, 1, -1)
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.linspace(np.min(feature_values), np.max(feature_values), 3)
            
            for threshold in thresholds:
                for direction in ['left', 'right']:
                    predictions = np.ones(n_samples)
                    if direction == 'left':
                        predictions[feature_values > threshold] = -1
                    else:
                        predictions[feature_values <= threshold] = -1
                    
                    misclassified = (predictions != y_ada)
                    weighted_error = np.sum(sample_weights[misclassified])
                    
                    if weighted_error < best_error:
                        best_error = weighted_error
                        self.feature_idx = feature_idx
                        self.threshold = threshold
                        self.direction = direction
        
        return best_error
    
    def fit(self, X, y, sample_weights):
        error = self.find_best_split(X, y, sample_weights)
        epsilon = 1e-10
        self.beta = 0.5 * np.log((1 - error + epsilon) / (error + epsilon))
        return self

class AdaBoost:
    def __init__(self, n_estimators=200):
        self.n_estimators = n_estimators
        self.stumps = []
        self.betas = []
        self.train_errors = []
        self.val_errors = []
        self.test_errors = []
        self.train_exp_loss = []
        self.val_exp_loss = []
        self.test_exp_loss = []
    
    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        n_samples = X_train.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        y_train_ada = np.where(y_train == 1, 1, -1)
        y_val_ada = np.where(y_val == 1, 1, -1)
        y_test_ada = np.where(y_test == 1, 1, -1)
        
        for t in tqdm(range(self.n_estimators), desc="Training AdaBoost"):
            stump = DecisionStump()
            stump.fit(X_train, y_train, sample_weights)
            predictions = stump.predict(X_train)
            
            misclassified = (predictions != y_train_ada)
            error = np.sum(sample_weights[misclassified])
            beta = 0.5 * np.log((1 - error + 1e-10) / (error + 1e-10))
            sample_weights *= np.exp(-beta * y_train_ada * predictions)
            sample_weights /= np.sum(sample_weights)
            
            #Store
            self.stumps.append(stump)
            self.betas.append(beta)
            
            #predictions for all sets
            train_pred = self._predict(X_train)
            val_pred = self._predict(X_val)
            test_pred = self._predict(X_test)
            
            self.train_errors.append(np.mean(np.sign(train_pred) != y_train_ada))
            self.val_errors.append(np.mean(np.sign(val_pred) != y_val_ada))
            self.test_errors.append(np.mean(np.sign(test_pred) != y_test_ada))
            
            self.train_exp_loss.append(np.mean(np.exp(-y_train_ada * train_pred)))
            self.val_exp_loss.append(np.mean(np.exp(-y_val_ada * val_pred)))
            self.test_exp_loss.append(np.mean(np.exp(-y_test_ada * test_pred)))
    
    def _predict(self, X):
        return np.sum([beta * stump.predict(X) for beta, stump in zip(self.betas, self.stumps)], axis=0)
    
    def predict(self, X):
        return np.sign(self._predict(X))
    
    def accuracy(self, X, y):
        y_ada = np.where(y == 1, 1, -1)
        return np.mean(self.predict(X) == y_ada)

def plot_results(adaboost):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(adaboost.train_errors, label='Train', color='blue')
    plt.plot(adaboost.val_errors, label='Validation', color='green')
    plt.plot(adaboost.test_errors, label='Test', color='red')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('0-1 Loss (Error Rate)')
    plt.title('Classification Error vs Boosting Rounds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(adaboost.train_errors, color='blue', label='Training Error')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Error Rate')
    plt.title('Training Error Progression')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(adaboost.train_exp_loss, label='Train', color='blue')
    plt.plot(adaboost.val_exp_loss, label='Validation', color='green')
    plt.plot(adaboost.test_exp_loss, label='Test', color='red')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Exponential Loss')
    plt.title('Exponential Loss vs Boosting Rounds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log') 
    
    plt.tight_layout()
    plt.show()

X_train, y_train, X_val, y_val, X_test, y_test = prepare_mnist_data()

adaboost = AdaBoost(n_estimators=200)
adaboost.fit(X_train, y_train, X_val, y_val, X_test, y_test)

final_train_acc = adaboost.accuracy(X_train, y_train)
final_val_acc = adaboost.accuracy(X_val, y_val)
final_test_acc = adaboost.accuracy(X_test, y_test)

print("\nFinal Accuracies:")
print(f"Training: {final_train_acc:.4f}")
print(f"Validation: {final_val_acc:.4f}")
print(f"Test: {final_test_acc:.4f}")

plot_results(adaboost)