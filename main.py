import matplotlib.pyplot as plt
import numpy as np
from Problem import Problem


def rhs(r, w):
    return np.sinh(r * 10)


multipoleIndex = 2
N = 10001
epsD = 1
r0 = .1
nu = 0.1
w = 1.04

p: Problem = Problem(N, multipoleIndex, nu, w, r0, epsD, rhs=rhs, Q0=0)

phi = p.getPhi()
rho = p.getRho()

r = np.linspace(0, 1, N)

# eps = 1 - 1. / w / (w + 1j*nu)
# phi_teor = -3. * epsD * r / (eps + 2 * epsD)


plt.figure(1)
plt.title(r"$\varphi$")
plt.plot(r, phi.real, "r")
plt.plot(r, phi.imag, "b")
# plt.plot(r, phi_teor.real, "k--")
# plt.plot(r, phi_teor.imag, "g--")
plt.grid()

plt.figure(2)
plt.title(r"$\rho$")
plt.plot(r, rho.real, "r")
plt.plot(r, rho.imag, "b")
plt.plot(r, rhs(r,w),'k--')
plt.grid()

plt.show()
