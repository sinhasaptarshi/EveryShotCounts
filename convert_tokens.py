import numpy as np
import os
from joblib import Parallel, delayed

T = 8
H = 14
W = 14


def reshape_tokens(path):
    for data in os.listdir(path):
        data = os.path.join(path, data)
        if not data.endswith('.npz'):
            print(data)
            continue
        z = np.load(data)['arr_0']
        if len(z.shape) == 5:
            continue
        reshape_z = z.transpose((0,2,1))
        N, C, THW = reshape_z.shape
        reshape_z = z.reshape(N, C, T, H, W)
        np.savez(data, reshape_z)

def main():
    paths = []
    for f in ['exemplar_tokens', 'saved_tokens']:
        for folder in ['train', 'val', 'test']:
            paths.append(os.path.join(f, folder))
    Parallel(n_jobs=6)(delayed(reshape_tokens)(path) for path in paths)

if __name__ == '__main__':
    main()
    
