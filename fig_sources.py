from NonlinearSources import NonlinearSources as ns
from ClusterParameters import ClusterParameters as cp
from scipy.special import eval_legendre as P
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

multipole_no = 0

Nr = 1000
Ntheta = 1000
Nxy = 1000

wp = 9 / 6.6e-16
nu = 0.03
Vf = 2.0e8
V0 = np.sqrt(3 / 5) * Vf
r0 = V0 / wp
epsInf = 3
a = 5e-7
alpha = r0 / a

epsD = 5.29
w2 = 0.5851

r = np.linspace(1e-3, 1, Nr)
theta = np.linspace(0, 2 * np.pi, Ntheta)

R, Theta = np.meshgrid(r, theta)
X, Y = np.meshgrid(np.linspace(-1, 1, Nxy), np.linspace(-1, 1, Nxy))

N_sp_grid_points = R.size
points = np.zeros((N_sp_grid_points, 2))
points[:, 0] = np.reshape((R * np.cos(Theta)), N_sp_grid_points)
points[:, 1] = np.reshape((R * np.sin(Theta)), N_sp_grid_points)

par = cp(nu, alpha, epsD, epsInf)
sources = ns(par, beta=0.1)

phiExt = np.reshape(sources.phiFunctions[multipole_no](R, w2) * P(multipole_no, Theta), (N_sp_grid_points))
rhoExt = sources.rhoFunctions[multipole_no](R, w2) * P(multipole_no, Theta)

phiExt_xy_grid = griddata(points, phiExt, (X, Y), method='cubic')

fig, ax = plt.subplots()
ax.contourf(X, Y, phiExt_xy_grid.real, 10)
plt.show()
