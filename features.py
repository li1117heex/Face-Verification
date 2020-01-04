import numpy as np

def get_pattern(block,block_size):
    pattern = np.zeros(8)
    feature = []
    for i in range(1, block_size[0] - 1):
        for j in range(1, block_size[1] - 1):
            pattern[0] = block[i - 1, j - 1] > block[i, j]
            pattern[1] = block[i - 1, j] > block[i, j]
            pattern[2] = block[i - 1, j + 1] > block[i, j]
            pattern[3] = block[i, j + 1] > block[i, j]
            pattern[4] = block[i + 1, j + 1] > block[i, j]
            pattern[5] = block[i + 1, j ] > block[i, j]
            pattern[6] = block[i + 1, j - 1] > block[i, j]
            pattern[7] = block[i, j - 1] > block[i, j]
            diff = np.array([pattern[i-1]!=pattern[i] for i in range(8)]).sum()
            if diff<=2:
                feature.append(np.dot(pattern,[2**i for i in range(8)]))
            else:
                feature.append(-1)
    return np.array(feature)

def lbp(input:np.ndarray,block_size=(10,10)):#int8 input with dim=2,edge around blocks is ignored
    uniform=[0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]
    if input.ndim!=2:
        print("input dim!=2")
    else:
        row_index = 0
        feature = []
        while row_index+block_size[0]<=input.shape[0]:
            column_index = 0
            while column_index+block_size[1]<=input.shape[1]:
                block = input[row_index:row_index+block_size[0],column_index:column_index+block_size[1]]
                pattern = get_pattern(block,block_size)
                feature.append(np.sum(pattern==-1))
                feature.extend([np.sum(pattern==i) for i in uniform])
                column_index+=block_size[1]
            row_index += block_size[0]
        return np.array(feature)
'''
img = cv2.imread('home.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT()
kp = sift.detect(gray,None)
kp,des = sift.compute(gray,kp)
'''
'''dence=cv2.FeatureDetector_create("Dense")
kp=dense.detect(gray)
kp,des=sift.compute(imgGray,kp)'''
#des is num of kp*128 matrix