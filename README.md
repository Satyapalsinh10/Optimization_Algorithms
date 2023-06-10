# Optimization_Techniques
Implementing  GD_SGD_BFGS_and_LBFGS Optimization


# Kernel Logistic Regression with L2 Regularizer

This repository contains an implementation of Kernel Logistic Regression with L2 regularizer using an empirical kernel map. The goal is to optimize the objective function J(ω) and obtain ω, which will be used to predict the labels of test data points.

## Problem Description

The objective function J(ω) is defined as follows:

J(ω) = -Σ log(σ(y_i ω > k_i)) + λω > ω,

where:
- k_i is a column vector such that k_i = [k(x_i, x_1), ..., k(x_i, x_j), ..., k(x_i, x_N)] >,
- y_i is the label of data point x_i,
- σ(v) = 1 / (1 + e^(-v)),
- N is the number of data points, and
- λ is the regularization parameter.

To compute p(y = 1|x) for a test data point x, the following steps are performed:
1. Compute k_x = [k(x, x_1), k(x, x_2), ..., k(x, x_N)] >, where x_i are the training data points.
2. Compute p(y = 1|x) = σ(ω > k_x).
3. If p(y = 1|x) > 0.5, the predicted label is 1; otherwise, it is -1.

The accuracy of the predictions will be reported.

The following optimization methods will be used to optimize J(ω):
a) GD (Gradient Descent)
b) SGD (Stochastic Gradient Descent) with two settings: p = 1 and p = 100
c) BFGS (Broyden-Fletcher-Goldfarb-Shanno) using a randomly sampled subset of 4000 training points
d) LBFGS (Limited-memory BFGS) using a small number of vectors to approximate the inverse Hessian

## Dataset

The dataset "data1.mat" will be used for the experiments.

## Usage

1. Clone the repository:

```
git clone https://github.com/your-username/kernel-logistic-regression.git
```

2. Install the necessary dependencies. Please refer to the `requirements.txt` file for the specific versions of the dependencies.

```
pip install -r requirements.txt
```

3. Run the main script to perform the experiments and compare the different optimization methods:

```
python main.py
```

4. Adjust the step sizes and other parameters in the script to experiment with different settings.

## Results

The main script will output the following information:
- The value of the cost function J(ω) at each iteration for each optimization method.
- The accuracy of the predictions for each optimization method.

The results will allow you to compare the performance of different optimization methods and observe how the value of the cost function decreases over time.

## Conclusion

By implementing Kernel Logistic Regression with L2 regularizer using empirical kernel mapping and comparing different optimization methods, this repository provides insights into the performance and accuracy of each method. Feel free to explore the code and experiment with different settings to further analyze and compare the methods.
