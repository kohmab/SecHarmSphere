from abc import abstractclassmethod
from abc import ABC
import re
import numpy as np

from ClusterParameters import ClusterParameters


class Oscillation(ABC):

    def __init__(self,
                 N: int,
                 m: int,
                 parameters: ClusterParameters) -> None:
        self.__nu = parameters.nu
        self.__multipoleNo = m
        self.__r0 = parameters.r0
        self.__epsD = parameters.epsD
        self.__epsInf = parameters.epsInf
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

    def __hash__(self) -> int:
        return
