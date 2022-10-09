from matplotlib.pyplot import disconnect
import numpy as np

def a(x):
    return 1 / (1 + np.exp(-x))


print(a(1.5))
print(a(1.75))
print(a(2.2))
print(a(2.45))
print()
print(a(0.3))
print(a(0.4))
print(a(0.6))
print(a(0.7))
print()
print(a(0.772))
print(a(0.79))
print(a(0.81))
print(a(0.814))
print()


dw = np.array([0.14*0.6, 0.14*0.6, -0.065*0.6, -0.064*0.6])
print(f"Delta next for neuron 1: {dw}")
fv = np.array([0.81*(1-0.81), 0.85*(1-0.85), 0.9*(1-0.9), 0.92*(1-0.92)])
dwfv = dw * fv
print("Deltas for neuron 1:", dwfv)
print()
input = np.array([[0,0,1],[0,0.5,1], [1,1,1], [1,1.5,1]])
dnext = np.array([0.01322769, 0.01117584, -0.00353118, -0.00285184])
s = dnext.T @ input
print(s)

