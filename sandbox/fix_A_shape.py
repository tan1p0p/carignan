import cv2
import h5py
from scipy.io  import loadmat

raw = 'data/raw/'
interim = 'data/interim/'
mouses = ['DM108', 'DM210', 'DM248']

for mouse in mouses:
    avi = f'{raw}{mouse}/{mouse}.avi'
    mat = f'{raw}{mouse}/{mouse}_A.mat'

    cap = cv2.VideoCapture(avi)
    _, frame = cap.read()
    d1, d2, _ = frame.shape

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
