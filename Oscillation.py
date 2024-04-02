from abc import abstractclassmethod
from abc import ABC
import re
import numpy as np


class Oscillation(ABC):

    def __init__(self,
                 N: int,
                 m: int,
                 nu: np.double, r0: np.double,
                 epsD: np.double = 1.,
                 epsInf: np.double = 1.) -> None:
        self.__nu = nu
        self.__multipoleNo = m
        self.__r0 = r0
        self.__epsD = epsD
        self.__epsInf = epsInf
        self.__r = np.linspace(0, 1, N)
        self._isFreqChanged = True
        self.__w = None

    @abstractclassmethod
    def getPhi(self, w):
        pass

    @abstractclassmethod
    def getRho(self, w):
        pass

    @abstractclassmethod
    def getPsi(self, w):
        pass

    @property
    def r(self):
        return self.__r

    @property
    def nu(self):
        return self.__nu

    @property
    def r0(self):
        return self.__r0

    @property
    def epsD(self):
        return self.__epsD

    @property
    def epsInf(self):
        return self.__epsInf

    @property
    def multipoleN0(self):
        return self.__multipoleNo

    @property
    def N(self):
        return self.__N

    @property
    def freq(self):
        return self.__w

    @freq.setter
    def freq(self, w):
        if self.__w != w:
            self._isFreqChanged = True
        self.__w = w
