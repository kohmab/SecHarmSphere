from typing import Callable
import numpy as np
from TDMAsolver import TDMAsolver as Solver
from Integrals import Coefficients


class Problem:

    __DIM: int = 2

    __multipoleNo: int

    __epsD: np.double

    __freq: np.double

    __nu: np.double

    __r0: np.double

    __kpaSq: np.complexfloating

    __Ac: np.ndarray

    __Bc: np.ndarray

    __Cc: np.ndarray

    __Av: np.ndarray

    __Bv: np.ndarray

    __Cv: np.ndarray

    __Fv: np.ndarray

    __rhsArr: np.ndarray

    __rhsFunc: Callable[[np.ndarray, np.double], np.ndarray]

    __solver: Solver

    __coef: Coefficients

    __solved: bool

    def _updateFreq(self):
        self.__solved = False

        self.__kpaSq = (self.__freq*(self.__freq + 1j *
                        self.__nu) - 1) / self.__r0 ** 2

        self.__solver.A = self.__Ac + self.__Av * self.__kpaSq
        self.__solver.B = self.__Bc + self.__Bv * self.__kpaSq
        self.__solver.C = self.__Cc - self.__Cv * self.__kpaSq

        self.__rhsArr = self.__rhsFunc(
            np.linspace(0, 1, self.__coef.N), self.__freq)

        self.__solver.F = np.zeros((self.__coef.N, self.__DIM), dtype=complex)

        self.__solver.F[1:-1, 0] = - self.__coef.alpha1[1:-1] * self.__rhsArr[0:-2] \
            - self.__coef.gamma1[1:-1] * self.__rhsArr[1:-1] \
            - self.__coef.beta1[1:-1] * self.__rhsArr[2:]
        self.__solver.F[0, 0] = - self.__coef.gamma1[0] * self.__rhsArr[0] \
            - self.__coef.beta1[0] * self.__rhsArr[1]
        self.__solver.F[-1, 0] = - self.__coef.gamma1[-1] * self.__rhsArr[-1] \
            - self.__coef.alpha1[-1] * self.__rhsArr[-2]

    def __init__(self, N: int, m: int,
                 nu: np.double, w: np.double,
                 r0: np.double, epsD: np.double,
                 rhs: Callable[[np.ndarray, np.double], np.ndarray]) -> None:
        """
            N - number of grid points
            m - multipole index
            nu - collisiton frequency [wp]
            w - initial frequency [wp]
            r0 - r0 = V0 / wp [a]
            epsD - permittivity of external media
            rhs - right hand side (source in the equation for rho). Function of r and freq
        """
        self.__coef = Coefficients(N)
        self.__solver = Solver(N, self.__DIM, dtype=complex)

        self.__multipoleNo = m
        self.__epsD = epsD
        self.__nu = nu
        self.__freq = w
        self.__r0 = r0

        self.__rhsFunc = rhs

        self.__Ac = np.zeros((N, self.__DIM, self.__DIM), dtype=complex)
        self.__Bc = np.zeros_like(self.__Ac, dtype=complex)
        self.__Cc = np.zeros_like(self.__Ac, dtype=complex)
        self.__Av = np.zeros_like(self.__Ac, dtype=complex)
        self.__Bv = np.zeros_like(self.__Ac, dtype=complex)
        self.__Cv = np.zeros_like(self.__Ac, dtype=complex)

        self.__Ac[:, 0, 0] = self.__coef.alpha2 - \
            m * (m+1) * self.__coef.alpha0
        self.__Ac[:, 1, 1] = self.__Ac[:, 0, 0]
        self.__Av[:, 0, 0] = self.__coef.alpha1

        self.__Bc[:, 0, 0] = self.__coef.beta2 - m * (m+1) * self.__coef.beta0
        self.__Bc[:, 1, 1] = self.__Bc[:, 0, 0]
        self.__Bv[:, 0, 0] = self.__coef.beta1

        self.__Cc[:, 0, 0] = self.__coef.gamma2 + \
            m * (m+1) * self.__coef.gamma0
        self.__Cc[:, 1, 1] = self.__Cc[:, 0, 0]
        self.__Cc[-1, 0, 1] -= 1. / (4 * np.pi * r0*r0) * epsD * (m + 1)
        self.__Cc[-1, 1, 1] += epsD * (m + 1)
        self.__Cv[:, 0, 0] = self.__coef.gamma1

        self._updateFreq()

    def setFreq(self, w: np.double) -> None:
        if self.__freq == w:
            return
        self.__freq = w
        self._updateFreq

    def _solve(self) -> None:
        if self.__solved:
            return
        self.__solver.solve()
        self.__solved = True

    def getPhi(self) -> np.ndarray:
        self._solve()
        return self.__solver.solution[:, 1]

    def getRho(self) -> np.ndarray:
        self._solve()
        return self.__solver.solution[:, 0]


if __name__ == "__main__":
    pass
    
