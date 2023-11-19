import matplotlib.pyplot as plt
import numpy as np
from Problem import Problem

def rhs(r, w):
    return np.ones_like(r)


multipoleIndex = 0
N = 100
epsD = 1
r0 = 100
nu = 0.1
w = 1

p : Problem = Problem(N, multipoleIndex, nu, w, r0, epsD, rhs)

phi = p.getPhi()
rho = p.getRho()

r = np.linspace(0,1,N)

plt.figure(1)
plt.title(r"$\varphi$")
plt.plot(r,phi.real,"r")
plt.plot(r,phi.imag,"b")
plt.grid()

plt.figure(2)
plt.title(r"$\rho$")
plt.plot(r,rho.real,"r")
plt.plot(r,rho.imag,"b")
plt.grid()

plt.show()