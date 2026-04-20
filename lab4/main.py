import matplotlib.pyplot as plt
import numpy as np


def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)


def dM_dt(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)


t0 = 1.0

exact_derivative = dM_dt(t0)

print("Крок 1")
print("Функція: M(t) = 50*e^(-0.1t) + 5*sin(t)")
print("Похідна: M'(t) = -5*e^(-0.1t) + 5*cos(t)")
print(f"t0 = {t0}")
print(f"Точне значення M'({t0}) = {exact_derivative:.10f}\n\n")


print("Крок 2\n")


hs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]

print(f"{'h':>12} {'D(h)':>18} {'Похибка |R(h)|':>20}")


results = []
for h in hs:
    D_h = (M(t0 + h) - M(t0 - h)) / (2 * h)
    error = abs(D_h - exact_derivative)
    results.append((h, D_h, error))

    print(f"{h:12.2e} {D_h:18.10f} {error:20.2e}")


min_error_idx = np.argmin([r[2] for r in results])
h_opt = results[min_error_idx][0]
D_opt = results[min_error_idx][1]
R_opt = results[min_error_idx][2]

print("\n")
print(f"Оптимальний крок h0 = {h_opt:.2e}")
print(f"Чисельне значення похідної при h0: D(h0) = {D_opt:.10f}")
print(f"Досягнута похибка R0 = {R_opt:.2e}")
print(f"Точне значення:  {exact_derivative:.10f}")

h = 0.001

print("Крок 3")
print(f"h = {h}")


print("Крок 4")


D_h = (M(t0 + h) - M(t0 - h)) / (2 * h)


D_2h = (M(t0 + 2 * h) - M(t0 - 2 * h)) / (4 * h)

print(f"Значення похідної з кроком h = {D_h:.10f}")
print(f"Значення похідної з кроком 2h = {D_2h:.10f}")
print(f"Точне значення = {exact_derivative:.10f}")


print("\n")
print("Крок 5")

R_h = abs(D_h - exact_derivative)

print(f"Значення похідної D(h) = {D_h:.10f}")
print(f"Точне значення = {exact_derivative:.10f}")
print(f"Похибка R_h = {R_h:.2e}")


print("\n")
print("Крок 6")


D_RR = D_h + (D_h - D_2h) / 3
R_RR = abs(D_RR - exact_derivative)

print(f"D(h) = {D_h:.10f}")
print(f"D(2h) = {D_2h:.10f}")
print(f"D_RR = {D_RR:.10f}")
print(f"Точне значення = {exact_derivative:.10f}")
print(f"Похибка R_RR = {R_RR:.2e}")

improvement = R_h / R_RR
print(f"\nПохибка зменшилась приблизно у {improvement:.1f} разів")


print("Крок 7")


D_h = (M(t0 + h) - M(t0 - h)) / (2 * h)
D_2h = (M(t0 + 2 * h) - M(t0 - 2 * h)) / (4 * h)
D_4h = (M(t0 + 4 * h) - M(t0 - 4 * h)) / (8 * h)

print(f"Значення з кроком h: D(h) = {D_h:.12f}")
print(f"Значення з кроком 2h: D(2h) = {D_2h:.12f}")
print(f"Значення з кроком 4h: D(4h) = {D_4h:.12f}")
print(f"Точне значення: exact = {exact_derivative:.12f}")


numerator = (D_2h - D_h) ** 2
denominator = D_4h - 2 * D_2h + D_h

D_Aitken = D_h - numerator / denominator if denominator != 0 else D_h


p = np.log(abs((D_4h - D_2h) / (D_2h - D_h))) / np.log(2)


R_Aitken = abs(D_Aitken - exact_derivative)

print("\n")
print(f"Уточнене значення за методом Ейткена: D* = {D_Aitken:.12f}")
print(f"Порядок точності оригінальної формули: p ≈ {p:.3f}")
print(f"Похибка після Ейткена: R3 = {R_Aitken:.2e}")

# Порівняння з попередніми похибками
print("\nПорівняння похибок:")
print(f"Похибка при h          : {abs(D_h - exact_derivative):.2e}")
print(f"Похибка після Рунге-Ромберга: {abs(D_RR - exact_derivative):.2e}")
print(f"Похибка після Ейткена : {R_Aitken:.2e}")
