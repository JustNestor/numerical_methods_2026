import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

os.makedirs("plots", exist_ok=True)


def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a = 0.0
b = 24.0


I0, _ = quad(f, a, b)
print(f"Точне значення інтегралу I0 = {I0:.15f}\n")


def simpson(N):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    I = h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1::2]) + 2 * np.sum(y[2:N:2]))
    return I


Ns = np.arange(10, 1001, 10)
eps = []
N_opt = None
eps_opt = float("inf")
I_opt = None


for N in Ns:
    I_N = simpson(N)
    e = abs(I_N - I0)
    eps.append(e)
    if e < eps_opt:
        eps_opt = e
        N_opt = N
        I_opt = I_N

print(f"N_opt (найкраща точність) = {N_opt}")
print(f"ε_PSOPT = {eps_opt:.2e}\n")


N0 = max(8, (N_opt // 10) // 8 * 8)
print(f"N0 = {N0}")

I_N0 = simpson(N0)
eps_PS0 = abs(I_N0 - I0)
print(f"ε_PS0 = {eps_PS0:.2e}\n")


I_N02 = simpson(N0 // 2)
I_R = I_N0 + (I_N0 - I_N02) / 15
eps_PSR = abs(I_R - I0)
print(f"ε_PSR (Рунге-Ромберга) = {eps_PSR:.2e}\n")


I_N04 = simpson(N0 // 4)
I_E = (I_N02**2 - I_N0 * I_N04) / (2 * I_N02 - I_N0 - I_N04)
eps_E = abs(I_E - I0)
p = np.log(abs((I_N02 - I_N0) / (I_N04 - I_N02))) / np.log(2)

print(f"I_E (Ейткена) = {I_E:.15f}")
print(f"ε_E (Ейткена) = {eps_E:.2e}")
print(f"Порядок методу p {p:.4f}\n")


plt.figure(figsize=(10, 6))
plt.plot(Ns, eps, "b-", linewidth=2)
plt.yscale("log")
plt.xlabel("Число розбиття N")
plt.ylabel("Похибка ε(N) = |I(N) - I₀|")
plt.title("Залежність похибки складової формули Сімпсона від N")
plt.grid(True, which="both", ls="--")
plt.savefig("plots/eps_vs_N.png", dpi=300, bbox_inches="tight")


def adaptive_simpson(a, b, tol, max_depth=30):
    def simpson_on_interval(x0, x2):
        x1 = (x0 + x2) / 2
        h = (x2 - x0) / 2
        return h / 3 * (f(x0) + 4 * f(x1) + f(x2))

    def recurse(x0, x2, I1, tol, depth):
        x1 = (x0 + x2) / 2
        I_left = simpson_on_interval(x0, x1)
        I_right = simpson_on_interval(x1, x2)
        I2 = I_left + I_right

        if abs(I2 - I1) <= 15 * tol or depth >= max_depth:
            return I2, 3  # 3 нові обчислення (x0, x1, x2 вже частково відомі)

        left, n1 = recurse(x0, x1, I_left, tol / 2, depth + 1)
        right, n2 = recurse(x1, x2, I_right, tol / 2, depth + 1)
        return left + right, n1 + n2 + 1

    I1 = simpson_on_interval(a, b)
    I_ad, calls = recurse(a, b, I1, tol, 0)
    total_calls = calls + 2
    return I_ad, total_calls


eps_list = [1e-6, 1e-8, 1e-10, 1e-12]
calls_list = []
real_eps_list = []

for tol in eps_list:
    I_ad, n_calls = adaptive_simpson(a, b, tol)
    real_eps = abs(I_ad - I0)
    calls_list.append(n_calls)
    real_eps_list.append(real_eps)
    print(
        f"Задана точність ε = {tol:.0e} | Реальна похибка = {real_eps:.2e} | Викликів f(x) = {n_calls}"
    )


plt.figure(figsize=(10, 6))
plt.plot(
    eps_list, calls_list, "ro-", linewidth=2, markersize=8, label="Адаптивний алгоритм"
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Задана точність ε")
plt.ylabel("Кількість обчислень f(x)")
plt.title("Ефективність адаптивного алгоритму")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig("plots/adaptive_calls_vs_eps.png", dpi=300, bbox_inches="tight")
