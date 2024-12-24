from os import path
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from ClusterParameters import ClusterParameters as ClusPar
from DipoleOscillation import DipoleOscillation as OscFh
from SecHarmOscillation import SecHarmOscillation as OcsSh
from freqFinder import FreqFinder as FrFnd
from Oscillation import Oscillation
from losses import getLosses, generateFilename
from fig_beta_intensity import calc_beta

import shelve

# Sodium cluster and field:
Vf = 1.07e8  # cm/s
wp_eV = 5.71  # eV
nu_eV = 0.0276  # eV
radius = 4e-7  # cm
epsInf = 1
field_intensity = 1e8  # W/cm^2

# Program parameters
epsDmin = 1
epsDmax = 2
NepsD = 200
Nw = 400
Nr = 250
#
hash_str = generateFilename(*[x for x in locals().values() if isinstance(x, float) or isinstance(x, int)])
#
colors = {0: '#77AB30', 1: '#000000', 2: '#D95A19'}
#

# Derived parameters
V0 = np.sqrt(3 / 5) * Vf
wp = wp_eV / 6.6e-16
r0 = V0 / wp
nu = nu_eV / wp_eV
nu_dip = nu + 3 / 4 * Vf / radius / wp
alpha = r0 / radius
beta = calc_beta(field_intensity, V0, radius)
#
epsD = np.linspace(epsDmin, epsDmax, NepsD)

par_dip = ClusPar(nu_dip, alpha, epsD[0], epsInf)

# Preparations
w0 = FrFnd(par_dip).getResonanceFrequencies(1, 1).real
w_range = w0 + np.linspace(-2, 2, Nw) * nu_dip
max_losses = np.zeros_like(epsD)
max_losses_freq = np.zeros_like(epsD)
mult_losses: List[np.ndarray] = [np.zeros_like(epsD) for i in range(3)]

losses: List[np.ndarray] = [np.zeros(Nw) for i in range(3)]

#
filename = Path(__file__).parent / "na_sphere_savedResults" / hash_str

if not filename.exists():
    for i, epsd in enumerate(epsD):
        par_dip = ClusPar(nu_dip, alpha, epsd, epsInf)
        par_sec_harm = ClusPar(nu, alpha, epsd, epsInf)

        # Oscillations
        oscillations: List[Oscillation] = []
        oscillations.append(OcsSh(Nr, 0, par_sec_harm, beta))
        oscillations.append(OscFh(Nr, par_dip))
        oscillations.append(OcsSh(Nr, 2, par_sec_harm, beta))

        for m, ocs in enumerate(oscillations):
            losses[m] = getLosses(ocs, w_range)

        total_losses = sum(losses)
        max_losses[i] = total_losses.real.max()

        max_losses_ind = np.argmax(total_losses)

        w0 = max_losses_freq[i] = w_range[max_losses_ind]
        w_range = w0 + np.linspace(-2, 2, Nw) * nu_dip

        for m in range(3):
            mult_losses[m][i] = losses[m][max_losses_ind]
    else:
        datafile = shelve.open(str(filename))
        datafile["max_losses"] = max_losses
        datafile["max_losses_freq"] = max_losses_freq
        datafile["mult_losses"] = mult_losses
else:
    datafile = shelve.open(str(filename))
    max_losses = datafile["max_losses"]
    max_losses_freq = datafile["max_losses_freq"]
    mult_losses = datafile["mult_losses"]

fig, ax = plt.subplots()
ax.plot(epsD, max_losses, label='max loss')
for m, losses in enumerate(mult_losses):
    ax.plot(epsD, losses, color=colors[m])
ax.grid()
plt.show()

print("done")
