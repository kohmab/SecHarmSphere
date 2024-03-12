import numpy as np
import scipy.special as ss
import scipy.optimize as so
import numba
from functools import cache
from sympy.functions.special.bessel import jn_zeros


class FreqFinder:
    """
        Class for finding resonance frequecnies of 
        spherical metallic nanoparticle placed 
        in media with permittivity epsD
    """
    @staticmethod
    @numba.njit
    def __kp(eps, r0):
        return np.sqrt(eps/(1.-eps) + 0j)/r0

    @staticmethod
    @numba.njit
    def __eps(w, nu):
        return 1 - 1./w/(w - 1j*nu)

    @staticmethod
    def __wFromEps(eps, nu):
        return np.sqrt(1./(1.-eps)-nu**2/4 + 0j) + 0.5j*nu

    @staticmethod
    def __epsFromKp(kp, r0):
        x = np.square(kp*r0)
        return x/(1+x)

    @staticmethod
    def __zeroFunction(n, eps, r0, epsD):
        kp = FreqFinder.__kp(eps, r0)
        j = ss.jve(n + 0.5, kp)
        djkp = n*j - ss.jve(n + 1.5, kp)*kp
        return np.real((n*eps + epsD*(n+1))*djkp + epsD*n*(n+1)*(eps - 1)*j)

    @staticmethod
    def __genGuessEps(n, N, r0, epsD):
        result = np.zeros(N)
        result[0] = -epsD*(n+1.)/n if n != 0 else np.NaN
        if N == 1:
            return result
        vals = np.array(jn_zeros(n+1, N-1))
        result[1:] = FreqFinder.__epsFromKp(vals, r0)
        return result

    def __init__(self, r0, nu, epsD, xtol=1e-9):
        """
            r0 -- charectiristic length of nonlocatity divided by the radius of the particle
            nu -- ratio of effective collision frequency and plasma frequency
            epsD -- permitivity of the surrounding media
        """

        if r0 > 1:
            raise Exception("Fermi radius is too big")
        self.__r0 = r0
        self.__nu = nu
        self.__epsD = epsD
        self.__xtol = xtol if xtol < 1e-9 else 1e-9

    def zeroFunc(self, n, eps):
        return FreqFinder.__zeroFunction(n, eps, self.__r0, self.__epsD)

    @cache
    def getResocnancePermittivities(self, n, Nz):
        """
            Returns the np.dnarray with values of nanoparticle dielectric function
            corresponding to the first Nz resonances of n-th multiplole mode of the nanopartilce.
            First element in array corresponds to the sufrace plasmon,
            subsequent ones correspond to the volume plasmons.
        """
        if Nz < 1:
            return np.array([], dtype=np.complex64)

        # Nz + 1 is need for np.isclose(spRootResult.root, 0) case
        guesses = FreqFinder.__genGuessEps(n, Nz+1, self.__r0, self.__epsD)

        if n == 0:
            return guesses[:-1]

        def F(x):
            return self.zeroFunc(n, x)

        def spFails():
            raise Exception(
                """Can not calculate resonance permittivity for surface plasmon""")

        result = np.zeros(Nz)

        # Surface plasmon:
        spGuess = guesses[0]
        spRootResult = so.root_scalar(F, x0=spGuess, xtol=self.__xtol)
        if not spRootResult.converged:
            spFails()

        if np.isclose(spRootResult.root, 0):
            spEpsInterval = np.linspace(0, guesses[1], 1000)
            zeroFuncValues = F(spEpsInterval)
            passThroughZero = zeroFuncValues[:-1]*zeroFuncValues[1:] < 0
            spGuess = spEpsInterval[np.where(passThroughZero)]
            spRootResult = so.root_scalar(F, x0=spGuess, xtol=self.__xtol)
            if not spRootResult.converged:
                spFails()

        result[0] = spRootResult.root

        # Volume plasmons
        for i, vpGuess in enumerate(guesses[1:-1]):
            vpRootResult = so.root_scalar(F, x0=vpGuess, xtol=self.__xtol)
            if not vpRootResult.converged:
                raise Exception(
                    f"""Can not calculate resonance permittivity for volume plasmon with No. {i+1}""")

            result[i+1] = vpRootResult.root
        return result

    def getResocnanceFrequencies(self, n, Nz):
        """
            Returns the np.dnarray with first Nz eigenfrequencies 
            of n-th multiplole mode of the nanopartilce.
            First element in array corresponds to the sufrace plasmon,
            subsequent ones correspond to the volume plasmons.
        """
        resEps = self.getResocnancePermittivities(n, Nz)
        return FreqFinder.__wFromEps(resEps, self.__nu)

    # def optFunc(self,  n, eps) :
    #     return FreqFinder.__abs2(self.zeroFunc(n, eps))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nu = 0.1
    r0 = .01

    ss.jn_zeros
    epsD = 1
    ff = FreqFinder(r0, nu, epsD)
    n = 21
    eps = np.linspace(-2, 1, 100000)
    Nz = 50

    zf = ff.zeroFunc(n, eps)
    zeros0 = FreqFinder._FreqFinder__genGuessEps(n, Nz, r0, epsD)
    zeros = ff.getResocnancePermittivities(n, Nz)
    print(ff.getResocnanceFrequencies(n, Nz))
    # def F(eps): return ff.optFunc(n, eps)
    # res = minimize(F, x0=-2)
    # print(res)
    fig, ax = plt.subplots()
    ax.plot(eps, zf, 'r')
    ax.scatter(zeros0, np.zeros(Nz), c="black")
    ax.scatter(zeros, np.zeros_like(zeros), c="red")
    ax.grid()
    plt.ylim([-1, 1])
    plt.show()
    # ax.plot(eps, of, 'k')
