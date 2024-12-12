from typing import List, Dict, Tuple, Any

from numpy import ndarray

from ClusterParameters import ClusterParameters
from DipoleOscillation import DipoleOscillation
from Oscillation import Oscillation
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle

from SecHarmOscillation import SecHarmOscillation
from freqFinder import FreqFinder

from losses import getLosses, generateFilename


class IterableParams:
    def __init__(self, params: Dict[str, np.ndarray]):
        self._params = params
        self._keys = list(self._params.keys())
        self._lens = [len(self._params[key]) for key in self._keys]
        self._current = [0] * len(self._keys)  # Initialize current indices

    def __iter__(self):
        self._current = [0] * len(self._keys)  # Reset current indices for a new iteration
        return self

    def __next__(self) -> Tuple[np.ndarray, ...]:
        if self._current is None:
            raise StopIteration

        result = tuple(self._params[key][self._current[i]] for i, key in enumerate(self._keys))

        # Update indices for the next iteration
        for i in reversed(range(len(self._current))):
            if self._current[i] + 1 < self._lens[i]:
                self._current[i] += 1
                break
            self._current[i] = 0
        else:
            self._current = None

        return result

nu = 0.02
a = 3.5e-7
beta = .1

useCache = True

iterParams = IterableParams({
    'wp': np.array([5, 6, 7, 8, 9, 10], dtype=np.float64) / 6.6e-16,
    'Vf': np.array([1, 1.2, 1.4, 1.6, 1.8, 2], dtype=np.float64) * 1e8,
    'epsInf': np.array([1, 2, 3, 4], dtype=np.float64)
})

wmin = 0.2
wmax = 1
Nw = 1000
w = np.linspace(wmin, wmax, Nw)

N = 250

for p in iterParams:

    wp, Vf, epsInf = p

    V0 = np.sqrt(3 / 5) * Vf
    r0 = V0 / wp
    alpha = r0 / a
    epsDk = list(np.linspace(1,3*epsInf/2,10))

    for epsD in epsDk:


        params = ClusterParameters(nu, alpha, epsD, epsInf)
        params_dip = ClusterParameters(nu + 3 / 4 * Vf / a / wp, alpha, epsD, epsInf)
        oscillations: List[Oscillation] = []
        oscillations.append(SecHarmOscillation(N, 0, params, beta))
        oscillations.append(DipoleOscillation(N, params_dip))
        oscillations.append(SecHarmOscillation(N, 2, params, beta))

        filename = "./savedResults/" + \
                   generateFilename(nu, alpha, epsD, epsInf, beta, N, Nw, wmin, wmax)
        if os.path.isfile(filename) and useCache:
            print("Loading saved results...")
            with open(filename, "rb") as handle:
                w, losses = pickle.load(handle)
        else:
            losses: List[np.ndarray] = []
            for osc in oscillations:
                losses.append(getLosses(osc, w))
            with open(filename, "wb") as handle:
                pickle.dump((w, losses), handle)

        colors = ["g", "b", "r"]
        fig = plt.figure(1)
        plt.clf()
        ff = FreqFinder(params)
        for i, l in enumerate(losses):
            plt.plot(w, l, colors[i])
            resFreq = ff.getResonanceFrequencies(
                oscillations[i].multipoleN0, 10).real
            if isinstance(oscillations[i], SecHarmOscillation):
                resFreq /= 2.
            plt.scatter(resFreq, np.zeros(10), c=colors[i], marker="x")
        plt.plot(w, sum(losses), 'y')
        plt.xlim((wmin, wmax))
        plt.grid()
        text = f"bt={beta}_wp={wp * 6.6e-16}ev_de={nu}_epsInf={epsInf}_Vf={Vf / 1e8}(1e8cms)_a={a * 1e7}nm_epsD={epsD}"
        plt.title(text)
        filename = f"./loss_res_curves_it/" + text + ".png"
        fig.savefig(filename)
        plt.pause(0.001)
        plt.draw()
