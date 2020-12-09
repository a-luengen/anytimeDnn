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

    def reset(self):
        self.dropCount = 0
        self.layerCount = 0

class ResNetDropRandNPolicy(ResnetDropResidualPolicy):

    name = 'drop-rand-n'

    def __init__(self, n):
        super(ResNetDropRandNPolicy, self).__init__()
        self._isMaxSet = False
        self._n = n
        self.skipConfigurationList = []

    def shouldDrop(self, idx=None) -> bool:
        if idx is not None:
            return self.skipConfigurationList[idx]

        # check predefined values
        self.dropCount += 1
        return self.skipConfigurationList[self.dropCount - 1]

    def setMaxSkipableLayers(self, maxCount: int):
        if self._n > maxCount:
            raise ValueError('Cannot skip more layers than available in this network.')

        super().setMaxSkipableLayers(maxCount)
        # random True or False array with max. N-True values
        self.skipConfigurationList = r_util.getRandomBoolListPermutation(maxCount, self._n)
        self._isMaxSet = True
    
    def getSkipConfigurationList(self)->List[bool]:
        return self.skipConfigurationList

class ResNetDropRandLastNPolicy(ResNetDropRandNPolicy):

    name = 'drop-last-rand-n'

    def __init__(self, n):
        super(ResNetDropRandLastNPolicy, self).__init__(n)
    
    def setMaxSkipableLayers(self, maxCount: int):        
        super().setMaxSkipableLayers(maxCount)

        self.skipConfigurationList = sorted(r_util.getRandomBoolListPermutation(maxCount, self._n))

class ResNetDropNRandNormalDistributionPolicy(ResNetDropRandNPolicy):
    
    name = 'drop-norm-n'

    def __init__(self, n):
        super(ResNetDropNRandNormalDistributionPolicy, self).__init__(n)
    
    def setMaxSkipableLayers(self, maxCount: int):
        super().setMaxSkipableLayers(maxCount)
        self.skipConfigurationList = r_util.getGaussDistributedBoolList(maxCount, self._n)
    

def setDropPolicy(policy: ResnetDropResidualPolicy) -> None:
    global RN_DROP_POLICY
    RN_DROP_POLICY['policy'] = policy

def getDropPolicy() -> ResnetDropResidualPolicy:
    return RN_DROP_POLICY['policy']



        