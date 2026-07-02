import numpy as np

# helper functions
def mean(X):
    mean = np.sum(X, axis=0) / X.shape[0]
    return mean

def variance(X):
    X_mean = mean(X)
    X_centered = X-X_mean
    variance = np.sum(X_centered ** 2, axis=0) / (X.shape[0]-1)
    return variance

def std(X):
    X_var = variance(X)
    std = np.sqrt(X_var)
    return std

def dot(u, v):
    assert u.shape == v.shape
    result = 0
    for i in range(u.shape[0]):
        result += u[i] * v[i]
    return result

def norm(v):
    squared_sum = 0
    for i in range(v.shape[0]):
        squared_sum += v[i] ** 2
    return squared_sum ** 0.5

def normalize(v):
    norm_v = norm(v)
    if norm_v < 1e-12:
        return np.zeros_like(v)
    return v / norm_v

def projection(vector, onto):
    if dot(onto, onto) < 1e-12:
        return np.zeros_like(vector)
    return dot(vector, onto) / dot(onto, onto) * onto

def classical_gram_schmidt(A):
    Q = [normalize(A[:, 0])]
    for i in range(1, A.shape[1]):
        a = A[:, i].astype(float)
        u = a.copy()
        for q in Q:
            u -= projection(a, q)
        q = normalize(u)
        Q.append(q)
    Q = np.column_stack(Q)
    R = Q.T @ A
    return Q, R
def modified_gram_schmidt(A):
    A_copy = A.astype(float).copy()
    Q = np.zeros(A.shape)
    R = np.zeros((A.shape[1], A.shape[1]))

    for i in range(0, A_copy.shape[1]):
        R[i, i] = norm(A_copy[:, i])
        if R[i, i] > 1e-12:
            Q[:, i] = A_copy[:, i] / R[i, i]
        else:
            R[i, i] = 0.0
            Q[:, i] = 0.0
        for j in range(i+1, A_copy.shape[1]):
            R[i, j] = np.dot(Q[:, i], A_copy[:, j])
            A_copy[:, j] = A_copy[:, j] - (R[i, j] * Q[:, i])
    return Q, R

# # Verifying for Gram-Schmidt
# A = np.array([[1, 2], [3, 4]])
# Q, R = modified_gram_schmidt(A)
# print(np.allclose(Q @ R, A))
#
# # Verifying Q orthogonality
# print(np.allclose(Q.T @ Q, np.eye(Q.shape[1])))
#
# # Verifying that R is triangular
# print(R)

# QR Algorithm using Classical Gram-Schmidt
def qr_algorithm(A, max_iter=1000, tol=1e-12):
    A = A.astype(float).copy()
    Q_total = np.eye(A.shape[0])
    for _ in range(max_iter):
        Q, R = modified_gram_schmidt(A)
        A = R @ Q
        Q_total = Q_total @ Q
        if np.all(np.abs(np.tril(A, k=-1)) < tol):
            break
    eigenvalues = np.diag(A)
    eigenvectors = Q_total
    return eigenvalues, eigenvectors

# PCA class
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
    def standardize(self, X):
        return (X - self.mean_)/np.where(self.std_ == 0, 1e-12, self.std_)
    def covariance_matrix(self, X_centered):
        cov_matrix = X_centered.T @ X_centered / (X_centered.shape[0]-1)
        return cov_matrix
    def eigen(self, A, max_iter=1000, tol=1e-12):
        return qr_algorithm(A, max_iter, tol)
    def select_components(self, eigenvalues, eigenvectors):
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues[:self.n_components], eigenvectors[:, :self.n_components]
    def project(self, X, W):
        return X @ W
    def fit_transform(self, X, max_iter=1000, tol=1e-12):
        self.mean_ = mean(X)
        self.std_ = std(X)

        X = self.standardize(X)
        cov_matrix = self.covariance_matrix(X)
        eigenvalues, eigenvectors = self.eigen(cov_matrix, max_iter, tol)
        top_eigenvalues, top_eigenvectors = self.select_components(eigenvalues, eigenvectors)
        self.eigenvalues_ = top_eigenvalues
        self.components_ = top_eigenvectors
        X_projected = self.project(X, top_eigenvectors)
        return X_projected
    def reconstruct(self, X_projected):
        return (X_projected @ self.components_.T) * np.where(self.std_ == 0, 1e-12, self.std_) + self.mean_
