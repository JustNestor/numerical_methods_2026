import numpy as np

np.random.seed(42)

n = 100
A = np.random.uniform(-10, 10, size=(n, n))

x_exact = np.full(n, 2.5)


b = A @ x_exact

np.savetxt("A.txt", A, fmt="%.10f")
np.savetxt("B.txt", b, fmt="%.10f")


def read_matrix(filename):
    return np.loadtxt(filename)


def read_vector(filename):
    return np.loadtxt(filename)


def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        U[i, i] = 1.0

    for k in range(n):
        for i in range(k, n):
            sum_lu = sum(L[i, j] * U[j, k] for j in range(k))
            L[i, k] = A[i, k] - sum_lu

        for i in range(k + 1, n):
            sum_lu = sum(L[k, j] * U[j, i] for j in range(k))
            U[k, i] = (A[k, i] - sum_lu) / L[k, k]

    return L, U


def solve_lu(L, U, b):
    n = b.shape[0]
    z = np.zeros(n)

    z[0] = b[0] / L[0, 0]
    for k in range(1, n):
        sum_l = sum(L[k, j] * z[j] for j in range(k))
        z[k] = (b[k] - sum_l) / L[k, k]

    x = np.zeros(n)
    x[n - 1] = z[n - 1]
    for k in range(n - 2, -1, -1):
        sum_u = sum(U[k, j] * x[j] for j in range(k + 1, n))
        x[k] = z[k] - sum_u

    return x


def matrix_vector_mult(A, x):
    return A @ x


def vector_norm(x):
    return np.max(np.abs(x))


A = read_matrix("A.txt")
b = read_vector("B.txt")


L, U = lu_decomposition(A)


np.savetxt("L.txt", L, fmt="%.10f")
np.savetxt("U.txt", U, fmt="%.10f")


x = solve_lu(L, U, b)


eps = 0.0
for i in range(n):
    s = sum(A[i, j] * x[j] for j in range(n))
    eps = max(eps, abs(s - b[i]))

print(f"Максимальна нев'язка ||Ax - b|| після LU = {eps:.2e}")


x0 = x.copy()
iter_count = 0
max_iter = 30

while iter_count < max_iter:
    iter_count += 1

    r = b - matrix_vector_mult(A, x0)
    dx = solve_lu(L, U, r)
    x_new = x0 + dx

    dx_norm = vector_norm(dx)
    residual_norm = vector_norm(matrix_vector_mult(A, x_new) - b)

    x0 = x_new.copy()

    if dx_norm <= 1e-14 and residual_norm <= 1e-14:
        break

print(f"Кількість ітерацій уточнення: {iter_count}")
print(f"Фінальна максимальна нев'язка: {residual_norm:.2e}")
