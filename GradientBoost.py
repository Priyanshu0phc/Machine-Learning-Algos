import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=100, noise_std=0.1, random_state=42):
    np.random.seed(random_state)
    x = np.random.uniform(0, 1, n_samples)
    y = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * x) + np.random.normal(0, noise_std, n_samples)

    split_idx = int(n_samples * 0.8)
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = generate_data(n_samples=100, noise_std=np.sqrt(0.01), random_state=42)

# Print dataset sizes
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

class DecisionStump:
    def __init__(self):
        self.threshold = None
        self.left_value = None
        self.right_value = None
        self.loss = float('inf')

    def fit(self, x, residuals, loss_fn='l2'):
        x = x.reshape(-1) 
        n_samples = x.shape[0]
        thresholds = np.linspace(0, 1, 20)

        for threshold in thresholds:
            left_mask = x <= threshold
            right_mask = ~left_mask

            if loss_fn == 'l2':
                left_val = np.mean(residuals[left_mask]) if np.sum(left_mask) > 0 else 0
                right_val = np.mean(residuals[right_mask]) if np.sum(right_mask) > 0 else 0
                current_loss = np.mean((residuals - np.where(left_mask, left_val, right_val)) ** 2)
            else:  # L1
                left_val = np.median(residuals[left_mask]) if np.sum(left_mask) > 0 else 0
                right_val = np.median(residuals[right_mask]) if np.sum(right_mask) > 0 else 0
                current_loss = np.mean(np.abs(residuals - np.where(left_mask, left_val, right_val)))

            if current_loss < self.loss:
                self.threshold = threshold
                self.left_value = left_val
                self.right_value = right_val
                self.loss = current_loss

    def predict(self, x):
        x = x.reshape(-1) #1d 
        return np.where(x <= self.threshold, self.left_value, self.right_value)

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.01, loss='l2'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.stumps = [] 
        self.train_loss = []
        self.initial_prediction = None 
    
    def fit(self, x_train, y_train):
        if self.loss == 'l2':
            self.F = np.full_like(y_train, np.mean(y_train))
            self.initial_prediction = np.mean(y_train)
        else:
            self.F = np.full_like(y_train, np.median(y_train))
            self.initial_prediction = np.median(y_train)
        
        for _ in range(self.n_estimators):
            residuals = self._compute_residuals(y_train, self.F)
            stump = DecisionStump()
            stump.fit(x_train, residuals, loss_fn=self.loss)
            self.stumps.append(stump)
            self.F += self.learning_rate * stump.predict(x_train)
            
            self._update_loss(y_train, self.F)
    
    def _compute_residuals(self, y, F):
        if self.loss == 'l2':
            return y - F
        else:
            return np.sign(y - F)  #l1 loss
    
    def _update_loss(self, y_true, F):
        if self.loss == 'l2':
            self.train_loss.append(np.mean((y_true - F) ** 2))  # MSE
        else:
            self.train_loss.append(np.mean(np.abs(y_true - F)))
    
    def predict(self, x):
        F = np.zeros(len(x))
        for stump in self.stumps:
            F += self.learning_rate * stump.predict(x)
        
        F += self.initial_prediction
        return F

def evaluate_gradient_boosting(model, x_train, y_train, x_test, y_test, n_plots=5):
    F_train = np.full_like(y_train, model.initial_prediction)
    F_test = np.full_like(y_test, model.initial_prediction)
    
    plot_iterations = np.linspace(0, model.n_estimators-1, n_plots, dtype=int)
    
    plt.figure(figsize=(15, 10))

    train_sort_idx = np.argsort(x_train)
    test_sort_idx = np.argsort(x_test)
    plt.subplot(2, 2, 1)
    plt.scatter(x_train, y_train, color='blue', alpha=0.4, label='True')
    for i, t in enumerate(plot_iterations):
        F_train = np.full_like(y_train, model.initial_prediction)
        for j in range(t+1):
            if j < len(model.stumps):
                F_train += model.learning_rate * model.stumps[j].predict(x_train)
        
        plt.plot(x_train[train_sort_idx], F_train[train_sort_idx], alpha=0.8, 
                label=f'Iter {t}' if i in [0, n_plots//2, n_plots-1] else "")
    
    plt.title(f'Training Predictions ({model.loss.upper()} Loss)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.subplot(2, 2, 2)
    plt.scatter(x_test, y_test, color='blue', alpha=0.4, label='True')
    for i, t in enumerate(plot_iterations):
        F_test = np.full_like(y_test, model.initial_prediction)
        for j in range(t+1):
            if j < len(model.stumps):
                F_test += model.learning_rate * model.stumps[j].predict(x_test)
        
        plt.plot(x_test[test_sort_idx], F_test[test_sort_idx], alpha=0.8, 
                label=f'Iter {t}' if i in [0, n_plots//2, n_plots-1] else "")
    
    plt.title(f'Test Predictions ({model.loss.upper()} Loss)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.subplot(2, 2, (3, 4))
    plt.plot(model.train_loss, color='red')
    plt.title(f'Training {model.loss.upper()} Loss vs Iterations')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('MSE' if model.loss == 'l2' else 'MAE')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    final_train_loss = model.train_loss[-1]
    final_pred_test = model.predict(x_test)
    if model.loss == 'l2':
        final_test_loss = np.mean((y_test - final_pred_test) ** 2)
    else:
        final_test_loss = np.mean(np.abs(y_test - final_pred_test))
    
    print(f"\nFinal Training {model.loss.upper()} Loss: {final_train_loss:.4f}")
    print(f"Final Test {model.loss.upper()} Loss: {final_test_loss:.4f}")

print("Gradient Boosting with L2 Loss (Squared Error):")
gb_l2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, loss='l2')
gb_l2.fit(x_train, y_train)
evaluate_gradient_boosting(gb_l2, x_train, y_train, x_test, y_test)

print("\nGradient Boosting with L1 Loss (Absolute Error):")
gb_l1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, loss='l1')
gb_l1.fit(x_train, y_train)
evaluate_gradient_boosting(gb_l1, x_train, y_train, x_test, y_test)