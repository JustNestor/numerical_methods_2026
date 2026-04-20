import numpy as np

np.random.seed(42)
n = 100

A = np.random.uniform(-5, 5, size=(n, n))
for i in range(n):
    off_diag = np.sum(np.abs(A[i])) - abs(A[i, i])
    A[i, i] = off_diag + np.random.uniform(8.0, 20.0)

x_exact = np.full(n, 2.5)
b = A @ x_exact

np.savetxt("A_lab8.txt", A, fmt="%.10f")
np.savetxt("B_lab8.txt", b, fmt="%.10f")


def read_matrix(name):
    return np.loadtxt(name)


def read_vector(name):
    return np.loadtxt(name)


def norm(x):
    return np.max(np.abs(x))


def simple_iteration(A, b, eps=1e-14, maxit=2000):
    n = len(b)
    x = np.ones(n)
    tau = 0.001
    C = np.eye(n) - tau * A
    d = tau * b
    for k in range(maxit):
        xnew = C @ x + d
        if norm(xnew - x) < eps:
            return xnew, k + 1
        x = xnew
    return x, maxit


def jacobi(A, b, eps=1e-14, maxit=2000):
    n = len(b)
    x = np.ones(n)
    Dinv = np.diag(1.0 / np.diag(A))
    R = A - np.diag(np.diag(A))
    for k in range(maxit):
        xnew = Dinv @ (b - R @ x)
        if norm(xnew - x) < eps:
            return xnew, k + 1
        x = xnew
    return x, maxit


def seidel(A, b, eps=1e-14, maxit=2000):
    n = len(b)
    x = np.ones(n)
    for k in range(maxit):
        xnew = x.copy()
        for i in range(n):
            s = 0.0
            for j in range(i):
                s += A[i, j] * xnew[j]
            for j in range(i + 1, n):
                s += A[i, j] * x[j]
            xnew[i] = (b[i] - s) / A[i, i]
        if norm(xnew - x) < eps:
            return xnew, k + 1
        x = xnew
    return x, maxit


A = read_matrix("A_lab8.txt")
b = read_vector("B_lab8.txt")


x1, it1 = simple_iteration(A, b)
print(f"Проста ітерація: {it1} ітерацій, нев'язка {norm(A @ x1 - b):.2e}")

x2, it2 = jacobi(A, b)
print(f"Метод Якобі: {it2} ітерацій, нев'язка {norm(A @ x2 - b):.2e}")

x3, it3 = seidel(A, b)
print(f"Метод Зейделя: {it3} ітерацій, нев'язка {norm(A @ x3 - b):.2e}")
