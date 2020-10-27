import logging
import random as rd
import resnet.utils as r_util
from typing import List

RN_DROP_POLICY = {
    'policy': None
}

class ResnetDropResidualPolicy(object):
    def __init__(self):
        self.dropCount = 0
        self.layerCount = 0
        self.maxSkipableLayers = 0

    def shouldDrop(self) -> bool:
        self.dropCount += 1
        return True

    def setMaxSkipableLayers(self, maxCount: int):
        self.maxSkipableLayers = maxCount

    def apply(self, residual_func, shortcut_func, x):
        self.layerCount += 1
        #logging.info(f"Called on {self.layerCount}-Layer")
        if self.shouldDrop():
            return shortcut_func(x)
        else:
            return residual_func(x) + shortcut_func(x)
        
    def reset(self):
        self.dropCount = 0
        self.layerCount = 0

class ResnetDropMaxRandomPolicy(ResnetDropResidualPolicy):
    def __init__(self, max_drop):
        super(ResnetDropMaxRandomPolicy, self).__init__()
        self._max = max_drop
    
    def setMaxSkipableLayers(self, maxCount: int):
        if maxCount < self._max:
            raise ValueError(f"maxCount of {maxCount} can not be smaller than {self._max}")
        
        super().setMaxSkipableLayers(maxCount)

    def shouldDrop(self) -> bool:
        if self.dropCount < self._max:
            drop = rd.random() >= 0.5
            if drop:
                self.dropCount += 1
            return drop
        return False

class ResNetDropRandNPolicy(ResnetDropResidualPolicy):
    def __init__(self, n):
        super(ResNetDropRandNPolicy, self).__init__()
        self._isMaxSet = False
        self._n = n
        self.shouldDropLayer = []

    def setMaxSkipableLayers(self, maxCount: int):
        if self._n > maxCount:
            raise ValueError('Cannot skip more layers than available in this network.')

        super().setMaxSkipableLayers(maxCount)
        # random True or False array with max. N-True values
        self.shouldDropLayer = r_util.getRandomBoolListPermutation(maxCount, self._n)
        self._isMaxSet = True

    def shouldDrop(self) -> bool:
        # check predefined values
        self.dropCount += 1
        return self.shouldDropLayer[self.dropCount - 1]
    
    def getDropConfig(self)->List[bool]:
        return self.shouldDropLayer

class ResNetDropRandLastNPolicy(ResNetDropRandNPolicy):

    def __init__(self, n):
        super(ResNetDropRandLastNPolicy, self).__init__(n)
    
    def setMaxSkipableLayers(self, maxCount: int):        
        super().setMaxSkipableLayers(maxCount)

        self.shouldDropLayer = sorted(r_util.getRandomBoolListPermutation(maxCount, self._n))

def setDropPolicy(policy: ResnetDropResidualPolicy) -> None:
    global RN_DROP_POLICY
    RN_DROP_POLICY['policy'] = policy

def getDropPolicy() -> ResnetDropResidualPolicy:
    return RN_DROP_POLICY['policy']



        