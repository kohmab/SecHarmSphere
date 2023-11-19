import matplotlib.pyplot as plt
import numpy as np
from Problem import Problem

def rhs(r, w):
    return np.ones_like(r) * kpaSq


multipoleIndex = 1
N = 10001
epsD = 1
r0 = .0001
nu = 0.1
w = 0.5

kpaSq = (w*(w+1j*nu) - 1) / r0 **2

p : Problem = Problem(N, multipoleIndex, nu, w, r0, epsD, rhs=None, Q0 = -1)

phi = p.getPhi()
rho = p.getRho()

r = np.linspace(0,1,N)

eps = 1 - 1./w/(w + 1j*nu)
phi_teor = -3. * epsD * r / (eps + 2 * epsD)


plt.figure(1)
plt.title(r"$\varphi$")
plt.plot(r,phi.real,"r")
plt.plot(r,phi.imag,"b")
plt.plot(r,phi_teor.real,"k--")
plt.plot(r,phi_teor.imag,"g--")
plt.grid()

plt.figure(2)
plt.title(r"$\rho$")
plt.plot(r,rho.real,"r")
plt.plot(r,rho.imag,"b")
plt.grid()

plt.show()