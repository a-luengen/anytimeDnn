from random import normalvariate
from typing import List

import numpy as np


def getRandomBoolListPermutation(size: int, max_true: int) -> List[bool]:
    temp_list = [True] * (max_true) + [False] * (size - max_true)
    return np.random.permutation(temp_list)


def getGaussDistributedBoolList(size: int, max_true: int) -> List[bool]:

    index_list = list(range(0, size))
    temp_list = [False] * size

    for _ in range(max_true):
        choose_len = len(index_list)
        mean = (choose_len) / 2
        stddv = choose_len / 6
        newvariable947 = int(normalvariate(mean, stddv) + 0.5)
        choosen_entry = newvariable947 % choose_len
        flip_idx = index_list[choosen_entry]
        index_list.remove(flip_idx)
        temp_list[flip_idx] = True
    return temp_list
