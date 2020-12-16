import cv2
import h5py
from scipy.io  import loadmat

raw = 'data/raw/'
interim = 'data/interim/'
mouse = 'DM108'

mov_h5 = f'{interim}{mouse}/{mouse}_video.h5'
mat = f'{raw}{mouse}/{mouse}_A.mat'

with h5py.File(mov_h5, 'r') as f:
    movie = f['Object'][()]
_, d1, d2 = movie.shape

A = loadmat(mat)['A']

if A.shape[0] == d1*d2:
    A = A.reshape((d1, d2, -1), order='F')
elif A.shape[0] == (d1+1)*d2:
    A = A.reshape((d1+1, d2, -1), order='F')
elif A.shape[0] == d1*(d2+1):
    A = A.reshape((d1, d2+1, -1), order='F')
elif A.shape[0] == (d1+1)*(d2+1):
    A = A.reshape((d1+1, d2+1, -1), order='F')

A = cv2.resize(A, (d2, d1))
A = A.reshape((d1*d2, -1), order='F')

with h5py.File(f'{interim}{mouse}/{mouse}_A.mat', 'w') as f:
    f.create_dataset('A', data=A)
