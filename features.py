import numpy as np
from collections import Counter

def get_pattern(block,block_size):
    pattern = np.zeros(8)
    feature = []
    for i in range(1, block_size[0] - 1):
        for j in range(1, block_size[1] - 1):
            pattern[0] = block[i - 1, j - 1] > block[i, j]
            pattern[1] = block[i - 1, j] > block[i, j]
            pattern[2] = block[i - 1, j + 1] > block[i, j]
            pattern[3] = block[i, j - 1] > block[i, j]
            pattern[4] = block[i, j + 1] > block[i, j]
            pattern[5] = block[i + 1, j - 1] > block[i, j]
            pattern[6] = block[i + 1, j] > block[i, j]
            pattern[7] = block[i + 1, j + 1] > block[i, j]
            diff = np.array([pattern[i-1]!=pattern[i] for i in range(8)]).sum()
            if diff<=2:
                feature.append(np.dot(pattern,[2**i for i in range(8)]))
            else:
                feature.append(None)
    return feature

def lbp(input:np.ndarray,block_size=(10,10)):#int8 input with dim=2,edge around blocks is ignored
    if input.ndim!=2:
        print("input dim!=2")
    else:
        row_index = 0
        column_index = 0
        feature = []
        while row_index+block_size[0]<=input.shape[0]:
            while column_index+block_size[1]<=input.shape[1]:
                block = input[row_index:row_index+block_size[0],column_index:column_index+block_size[1]]
                feature.append(Counter(get_pattern(block,block_size)).values())
        return np.concatenate(feature,axis=0)