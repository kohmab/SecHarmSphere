from typing import List
from ClusterParameters import ClusterParameters
from DipoleOscillation import DipoleOscillation
from Oscillation import Oscillation
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from SecHarmOscillation import SecHarmOscillation


def getLossesAtOneFreq(osc: Oscillation, w: float):
    psi = osc.getPsi(w)
    rhoConj = osc.getRho(w).conj()
    rsq = np.square(osc.r)
    # @TODO calculate with Integrals
    integralOverR = np.trapz(osc.r, psi*rhoConj*rsq)
    integralOverTheta = 2./(osc.multipoleN0*2 + 1)
    integralOverPhi = 2*np.pi
    coef = -osc.nu/2*(osc.freq/(osc.freq + 1j*osc.nu))
    return np.real(coef*integralOverPhi*integralOverTheta*integralOverR)


def getLosses(osc: Oscillation, w: np.ndarray) -> np.ndarray:
    result = np.zeros_like(w)
    print(f"Calculating losses for multipole #{osc.multipoleN0}...")
    for i, freq in tqdm(enumerate(w)):
        result[i] = getLossesAtOneFreq(osc, freq)
    return result


if __name__ == "__main__":
    nu = 0.01
    r0 = 0.1
    epsD = 1.
    epsInf = 1.
    beta = 0.01
    N = 1001

    wmin = 0.2
    wmax = 1.1
    Nw = 1000
    w = np.linspace(wmin, wmax, Nw)

    params = ClusterParameters(nu, r0, epsD, epsInf)
    oscillations: List[Oscillation] = []
    oscillations.append(SecHarmOscillation(N, 0, params, beta))
    oscillations.append(DipoleOscillation(N, params))
    oscillations.append(SecHarmOscillation(N, 2, params, beta))
    losses: List[np.ndarray] = []
    for osc in oscillations:
        losses.append(getLosses(osc, w))

    plt.figure(1)
    for i in range(3):
        plt.plot(w, losses[i])
    plt.grid()
    plt.show()
