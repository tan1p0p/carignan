import os

import h5py
from scipy.io import loadmat

f = loadmat('data/raw/ABNs_patterns.mat')

for i, mouse in enumerate(['DM108', 'DM210', 'DM248']):
    save_dir = f'data/interim/{mouse}/'
    try:
        os.makedirs(save_dir)
    except:
        pass

    for j in range(9):
        x = f['X'][i, j]
        w = f['W'][i, j]
        h = f['H'][i, j]
        with h5py.File(f'{save_dir}{mouse}_sync-{j}.mat', 'w') as w_f:
            w_f.create_dataset('X', data=x)
            w_f.create_dataset('W', data=w)
            w_f.create_dataset('H', data=h)
        # savemat(f'{save_dir}{mouse}_sync-{j}.mat', {'X': x, 'W': w, 'H': h})
