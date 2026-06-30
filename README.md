# Machine Learning Mathematics from Scratch:
### PCA, Neural Networks, and Naive Bayes
**by Jiho Lee**

Implementing key mathematical concepts from scratch to consolidate my understanding of the [Mathematics for Machine Learning and Data Science Specialization](https://deeplearning.ai) by DeepLearning.ai.

Modern machine learning libraries hide much of the underlying mathematical operations behind high-level APIs. While this abstraction makes development efficient, it can make it difficult to understand what is happening internally. By rebuilding these core algorithms from scratch, this project aims to develop an intuitive understanding of the mathematical foundations and mechanisms behind machine learning algorithms.

No high-level machine learning frameworks (such as PyTorch, TensorFlow, or scikit-learn) are used for model implementation. The goal is to implement the fundamental algorithms manually using only NumPy (TensorFlow was only used for loading the MNIST data).

## Projects
* **Course 1: Linear Algebra** $\rightarrow$ **Principal Component Analysis (PCA)**
  * Implementing:
    * Helper Functions: Mean, Variance, Standard Deviation, Covariance, Dot Product, Euclidean Norm, Vector Normalization, Vector Projection
    * Modified Gram-Schmidt orthogonalization
    * QR Decomposition
    * QR Algorithm for eigendecomposition
    * Principal Component Analysis
      1. Standardize Data
      2. Compute Covariance Matrix
      3. Eigendecomposition of the covariance matrix using the QR algorithm
      4. Projecting the original data on the principal components
* **Course 2: Calculus** $\rightarrow$ **Neural Network**
  * Implementing
    * Forward propagation
    * Backpropagation
    * Gradient descent
    * Prediction
* **Course 3: Probability** $\rightarrow$ **Naive Bayes**
  * Implementing:
    * Data preprocessing
    * Probability estimation
    * Prior and likelihood calculations
    * Bayes' Theorem
    * Classification

## Evaluation Plan
I will use the **MNIST Handwritten Digits** dataset to test and compare the models:
1. **Dimensionality Reduction:** Apply my PCA implementation and test reconstruction.
2. **Neural Network Learning:** Train my Neural Network implementation.
3. **Probabilistic Learning:** Train my Naive Bayes implementation.
4. **Comparison:** Evaluate the performance, and efficiency of the models(neural network, naive bayes').

## Results
* To be added

## Future Extensions
* Implementing industry-standard optimizers such as Adam, Momentum
* Implementing Convolutional or Recurrent NNs
* Building a small automatic differentiation engine
