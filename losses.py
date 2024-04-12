from typing import List
from ClusterParameters import ClusterParameters
from DipoleOscillation import DipoleOscillation
from Oscillation import Oscillation
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path
import pickle

from SecHarmOscillation import SecHarmOscillation
from freqFinder import FreqFinder


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


def generateFilename(*args) -> str:
    res = 0
    for arg in args:
        res += res*31 + hash(arg)
    return hex(res)


if __name__ == "__main__":
    nu = 0.01
    r0 = 0.05
    epsD = 3
    epsInf = 2
    beta = 1.2
    N = 751

    wmin = 0.1
    wmax = 0.45
    Nw = 1000
    w = np.linspace(wmin, wmax, Nw)

    params = ClusterParameters(nu, r0, epsD, epsInf)
    oscillations: List[Oscillation] = []
    oscillations.append(SecHarmOscillation(N, 0, params, beta))
    oscillations.append(DipoleOscillation(N, params))
    oscillations.append(SecHarmOscillation(N, 2, params, beta))

    filename = "./savedResults/" + \
        generateFilename(nu, r0, epsD, epsInf, beta, N, Nw, wmin, wmax)
    if not os.path.isfile(filename):
        losses: List[np.ndarray] = []
        for osc in oscillations:
            losses.append(getLosses(osc, w))
        with open(filename, "wb") as handle:
            pickle.dump((w, losses), handle)
    else:
        print("Loading saved results...")
        with open(filename, "rb") as handle:
            w, losses = pickle.load(handle)

    colors = ["g", "r", "b"]
    plt.figure(1)
    ff = FreqFinder(params)
    for i, l in enumerate(losses):
        plt.plot(w, l, colors[i])
        resFreq = ff.getResocnanceFrequencies(
            oscillations[i].multipoleN0, 10).real
        if isinstance(oscillations[i], SecHarmOscillation):
            resFreq /= 2.
        plt.scatter(resFreq, np.zeros(10), c=colors[i], marker="x")
    plt.plot(w, sum(losses), 'y')
    plt.xlim((wmin, wmax))
    plt.grid()
    plt.show()
