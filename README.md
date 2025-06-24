Machine Learning Algorithms: From Ground Up

This repository hosts a series of core Machine Learning methods implemented entirely from scratch to illustrate fundamental concepts and internal mechanics—no black-box libraries here.

Contents

digit_detection.py — PCA & FDA for feature reduction, followed by discriminant analysis on MNIST digits.

decision_tree.ipynb — Manual decision tree, bagging ensemble, and random forest using Gini impurity.

regression.ipynb — Polynomial curve fitting and sine approximation experiments.

Adaboost.py — Custom AdaBoost with decision stumps on PCA-preprocessed MNIST (digits 0 vs 1).

GradientBoost.py — Gradient boosting regressor built with stump learners, supporting L2 and L1 losses.

NeuralNetwork.py — Minimal 2-feature, 1-hidden-neuron binary classifier trained via backpropagation.

README.md — This overview.

Quickstart

Install required packages:

pip install numpy scipy matplotlib scikit-learn

Run Python scripts:

python digit_detection.py
python Adaboost.py
python GradientBoost.py
python NeuralNetwork.py

Launch notebooks:

jupyter notebook decision_tree.ipynb
jupyter notebook regression.ipynb

Project Overviews

1. Digit Detection (digit_detection.py)

Applies PCA and Fisher Discriminant Analysis to compress MNIST images into low-dimensional spaces, then classifies digits using linear and quadratic discriminant analysis. Generates performance metrics and visual comparisons of each reduction technique.

2. Decision Tree & Ensembles (decision_tree.ipynb)

Constructs a binary decision tree by optimizing Gini impurity, then demonstrates bagging and a random forest ensemble. Compares error rates and feature importance across models with interactive plots.

3. Polynomial Regression & Sine Fit (regression.ipynb)

Fits polynomials of various degrees (Taylor and least squares approaches) to approximate the sine curve. Analyzes error behavior (MSE and peak deviation) against polynomial order to determine the ideal model complexity.

4. AdaBoost with Stumps (Adaboost.py)

From-scratch implementation of AdaBoost employing simple threshold-based decision stumps over PCA-transformed MNIST for binary classification (0 vs 1). Visualizes weight evolution and ensemble accuracy over boosting rounds.

5. Gradient Boosting (GradientBoost.py)

Implements gradient boosting regression by iteratively fitting decision stumps to residuals. Supports both squared-error and absolute-error losses, and plots training loss curves to highlight robustness differences.

6. Simple Neural Network (NeuralNetwork.py)

A toy feedforward network: two inputs → one sigmoid-activated hidden node → linear output. Learns a synthetic binary dataset via gradient descent, with loss trajectory and boundary plot to illustrate training dynamics.
