import enum
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
    integralOverR = np.trapz(psi*rhoConj*rsq, osc.r)
    integralOverTheta = 2./(osc.multipoleN0*2 + 1)
    integralOverPhi = 2*np.pi
    coef = osc.nu*osc.freq/(osc.freq + 1j*osc.nu)/2
    return np.real(coef*integralOverPhi*integralOverTheta*integralOverR)


def getLosses(osc: Oscillation, w: np.ndarray) -> np.ndarray:
    result = np.zeros_like(w)
    print(f"Calculating losses for multipole #{osc.multipoleN0}...")
    total = w.size
    for i, freq in tqdm(enumerate(w), total=total):
        result[i] = getLossesAtOneFreq(osc, freq)
    return result


if __name__ == "__main__":
    nu = 0.012
    r0 = 0.078
    epsD = 3./2.
    epsInf = 1.
    beta = 1.2
    N = 401

    wmin = 0.3
    wmax = 1
    Nw = 1500
    w = np.linspace(wmin, wmax, Nw)

    params = ClusterParameters(nu, r0, epsD, epsInf)
    oscillations: List[Oscillation] = []
    oscillations.append(SecHarmOscillation(N, 0, params, beta))
    oscillations.append(DipoleOscillation(N, params))
    oscillations.append(SecHarmOscillation(N, 2, params, beta))
    losses: List[np.ndarray] = []
    for osc in oscillations:
        losses.append(getLosses(osc, w))

    colors = ["g", "r", "b"]
    plt.figure(1)
    for i, l in enumerate(losses):
        plt.plot(w, l, colors[i])
    plt.grid()
    plt.show()
