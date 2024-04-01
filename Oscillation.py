from abc import abstractclassmethod


class Oscillation():

    @abstractclassmethod
    def getPhi(self, w):
        pass

    @abstractclassmethod
    def getRho(self, w):
        pass

    @abstractclassmethod
    def getKsi(self, w):
        pass

    @abstractclassmethod
    def getR(self):
        pass

    @abstractclassmethod
    def getMultipoleIndex(self):
        pass
