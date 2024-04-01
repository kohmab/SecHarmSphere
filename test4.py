from DipoleOscillation import DipoleOscillation
from Problem import Problem
import matplotlib.pyplot as plt


multipoleIndex = 1
N = 1001
epsD = 1
epsInf = 10
r0 = 0.1
nu = 1
w = 1.5

p = Problem(N, multipoleIndex, nu, w, r0, epsD=epsD, epsInf=epsInf, Q0=-1.)
ansol = DipoleOscillation(N, nu, r0, epsD=epsD, epsInf=epsInf)

rho_num = p.getRho()
rho_an = ansol.getRho(w)

phi_num = p.getPhi()
phi_an = ansol.getPhi(w)
r = p.getR()

plt.figure(1)
plt.plot(r, rho_num.real, "r")
plt.plot(r, rho_an.real, "k:")
plt.plot(r, rho_num.imag, "b")
plt.plot(r, rho_an.imag, "k:")
plt.grid()

plt.figure(2)
plt.plot(r, phi_num.real, "r")
plt.plot(r, phi_an.real, "k:")
plt.plot(r, phi_num.imag, "b")
plt.plot(r, phi_an.imag, "k:")
plt.grid()
plt.show()
