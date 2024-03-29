import matplotlib.pyplot as plt
import numpy as np
from Integrals import Coefficients
from Problem import Problem

from scipy.special import spherical_jn


def rhsPhi(r, w):
    return A1*G(k1, r)


multipoleIndex = 0
N = 1001
epsD = 1000.
epsInf = 1
r0 = .04
nu = 5
w = 0.5
k1 = 0.1
A1 = 1.
Q = 0


def G(k, r):
    return (
        spherical_jn(multipoleIndex, k * r) / k /
        spherical_jn(multipoleIndex, k, True)
    )


p: Problem = Problem(N, multipoleIndex, nu, w, r0,
                     epsD, epsInf=epsInf, rhsrho=None, rhsphi=rhsPhi, Q0=Q)

phi = p.getPhi()
rho = p.getRho()

r = p.getR()


# theor sol
n = multipoleIndex
kp = np.sqrt(w * (w + 1j * nu) - 1./epsInf) / r0

al = A1*k1**2/(kp**2-k1**2)/4./np.pi
gm = 1./4/np.pi/epsInf/kp**2
e0 = epsD/epsInf
M = np.eye(2, dtype=complex)
D = np.ones(2, dtype=complex)
M[0, 0] = n
M[0, 1] = gm+4*np.pi*r0**2
M[1, 0] = n+e0*(n+1)
M[1, 1] = gm*(1+e0*(n+1)*G(kp, 1))
D[0] = -gm*al-4*np.pi*r0**2*al-r0**2*A1
D[1] = -gm*al+e0*((2*n+1)*Q-(n+1)*gm*(al+A1/4/np.pi)*G(k1, 1))
ca = np.linalg.solve(M, D)
C = ca[0]
A = ca[1]
print(M@ca-D)


rho_teor = A*G(kp, r)+al*G(k1, r)
phi_teor = C*r**n + gm*(rho_teor+A1*G(k1, r)/4/np.pi)

plt.figure(1)
plt.title(r"$\varphi$")
plt.plot(r, phi.real, "r")
plt.plot(r, phi.imag, "b")
plt.plot(r, phi_teor.real, "k:")
plt.plot(r, phi_teor.imag, "k:")
plt.grid()

plt.figure(2)
plt.title(r"$\rho$")
plt.plot(r, rho.real, "r")
plt.plot(r, rho.imag, "b")
plt.plot(r, rho_teor.real, "k:")
plt.plot(r, rho_teor.imag, "k:")
plt.grid()

plt.show()
