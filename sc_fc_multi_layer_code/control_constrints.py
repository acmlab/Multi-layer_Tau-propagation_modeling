"""
LQR utilities for control design with torch backend.
Includes:
- LQR gain computation
- Custom Riccati equation solver (stable)
- Controllability matrix utilities
"""

import torch


def lqr(A, B, Q, R):
    """
    Solve the continuous-time LQR controller:
        dx/dt = A x + B u
        cost = integral x.T*Q*x + u.T*R*u

    Returns:
        K (torch.Tensor): Optimal feedback gain
        X (torch.Tensor): Solution to Riccati equation
        eigVals (torch.Tensor): Eigenvalues of closed-loop system
    """
    X = solve_continuous_are(A, B, Q, R)
    K = torch.linalg.solve(R, B.T @ X)
    eigVals = torch.linalg.eigvals(A - B @ K)
    return K, X, eigVals


def solve_continuous_are(A, B, Q, R, epsilon=1e-8, dtype=torch.float64):
    """
    Solve the continuous-time Algebraic Riccati Equation using Hamiltonian approach.

    Args:
        A, B, Q, R (torch.Tensor): System matrices
        epsilon (float): Regularization threshold
        dtype (torch.dtype): Precision used (default: float64)

    Returns:
        X (torch.Tensor): Solution to Riccati equation
    """
    device = A.device
    A, B, Q, R = [mat.to(dtype=dtype) for mat in (A, B, Q, R)]
    R = R + epsilon * torch.eye(R.size(0), device=device, dtype=dtype)

    n = A.shape[0]
    H = torch.zeros((2 * n, 2 * n), dtype=dtype, device=device)
    H[:n, :n] = A
    H[:n, n:] = -B @ torch.linalg.solve(R, B.T)
    H[n:, :n] = -Q
    H[n:, n:] = -A.T

    eigenvalues, eigenvectors = torch.linalg.eigh(H)
    neg_indices = eigenvalues < -epsilon
    P = eigenvectors[:, neg_indices]
    X = P[:n, :].T @ P[n:, :]
    return X.to(dtype=torch.float32)


def ctrb(A, B):
    """
    Compute the controllability matrix.

    Args:
        A (torch.Tensor): System matrix [n x n]
        B (torch.Tensor): Input matrix [n x m]

    Returns:
        torch.Tensor: Controllability matrix [n x (n*m)]
    """
    n = A.size(0)
    ctrb_mat = B
    for i in range(1, n):
        ctrb_mat = torch.cat((ctrb_mat, A**i @ B), dim=1)
    return ctrb_mat


def new_ctrb(A, B):
    """
    Stable version of controllability matrix computation.

    Args:
        A (torch.Tensor): System matrix [n x n]
        B (torch.Tensor): Input matrix [n x m]

    Returns:
        torch.Tensor: Controllability matrix [n x (n*m)]
    """
    n, m = A.size(0), B.size(1)
    ctrb_matrix = torch.zeros((n, n*m), device=A.device, dtype=A.dtype)

    A_stable = A.clone()
    eigenvalues = torch.linalg.eigvals(A_stable)
    max_eig = torch.max(torch.abs(eigenvalues))
    if max_eig > 1.0:
        A_stable = A_stable / (max_eig + 0.01)

    current_block = B.clone()
    ctrb_matrix[:, :m] = current_block

    for i in range(1, n):
        current_block = A_stable @ current_block
        ctrb_matrix[:, i * m:(i + 1) * m] = current_block

    return ctrb_matrix
