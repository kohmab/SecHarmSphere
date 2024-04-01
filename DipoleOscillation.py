import numpy as np
from scipy.special import spherical_jn, jve
from Oscillation import Oscillation


class DipoleOscillation(Oscillation):
    def __init__(self,
                 N: int,
                 nu: np.double, r0: np.double,
                 epsD: np.double = 1.,
                 epsInf: np.double = 1.) -> None:
        self.__nu = nu
        self.__r0 = r0
        self.__epsD = epsD
        self.__epsInf = epsInf
        self.__r = np.linspace(0, 1, N)

    def __radFunc(self, r):
        return spherical_jn(1, self.__kp*r)/self.__kp/spherical_jn(1, self.__kp, True)

    def __updateFreq(self, w):
        self.__kp = np.sqrt(w*(w+1j*self.__nu) - 1. /
                            self.__epsInf) / self.__r0
        self.__eps = self.__epsInf - 1. / w/(w + 1j * self.__nu)

        v0 = jve(1.5, self.__kp)
        v1 = jve(2.5, self.__kp)
        self.__G = v0 / (v0 - v1*self.__kp)

        denum = self.__eps + 2 * self.__epsD * \
            (1 + (self.__eps - self.__epsInf)/self.__epsInf*self.__G)
        self.__C = -3 * self.__epsD/denum

    def getPhi(self, w):
        self.__updateFreq(w)
        return self.__C*(self.__r - self.__radFunc(self.__r)/w/(w + 1j * self.__nu)/self.__epsInf)

    def getRho(self, w):
        self.__updateFreq(w)
        return -self.__C*self.__eps/(4.*np.pi*self.__r0**2 * self.__epsInf)*self.__radFunc(self.__r)

    def getPsi(self, w):
        phi = self.getPhi()
        rho = self.getRho()
        return phi + 4*np.pi*self.__r0**2 * rho

    def getR(self):
        return self.__r

    def getMultipoleIndex(self):
        pass

    def getRhoExtMono(self):
        pass

    def getRhoExtQuadro(self):
        pass

    def getPhiExtMono(self):
        pass

    def getPhiExtQuadro(self):
        pass
