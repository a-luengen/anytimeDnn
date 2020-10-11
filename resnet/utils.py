import random
import numpy as np
from typing import List

def getRandomBoolListPermutation(size: int, max_true: int) -> List[bool]:
    temp_list = [True] * (max_true) + [False] * (size - max_true)
    return np.random.permutation(temp_list)