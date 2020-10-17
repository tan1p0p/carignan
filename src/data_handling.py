import json
import os
import time

import cv2
import h5py
import numpy as np
from scipy.sparse import csc_matrix

from caiman.source_extraction.cnmf.estimates import Estimates


def ready_mov_file(mov_path, mov_key, end_frame, first_frame=0, return_mov_data=False):
    if os.path.splitext(mov_path)[-1] == '.avi':
        cap = cv2.VideoCapture(mov_path)
        dims = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame > frame_count:
            raise ValueError(f'end frame ({end_frame}) > whole video frame ({frame_count})!')

        mov_data = np.zeros((end_frame - first_frame,) + dims)
        for i in range(first_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, frame = cap.read()
            mov_data[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f'{mov_path} succesfully loaded!')

    else:
        with h5py.File(mov_path, 'r') as f:
            print('keys:', f.keys())
            if len(f[mov_key].shape) == 3:
                mov_data = f[mov_key][first_frame:end_frame]
            elif len(f[mov_key].shape) == 4:
                mov_data = f[mov_key][0, first_frame:end_frame]

        dims = (mov_data.shape[1], mov_data.shape[2])

    root, sample = tuple(os.path.dirname(mov_path).split('data/raw/'))
    dir_name = os.path.join(root, 'data/interim/', sample)
    # out_path = os.path.join(dir_name, f'{first_frame}_{end_frame}.h5')
    out_path = os.path.join(dir_name, f'{time.time_ns()}.h5')
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('mov', data=mov_data)

    if return_mov_data:
        return out_path, dims, mov_data
    else:
        del mov_data
        return out_path, dims


def load_annotation_file(file_path, end_frame, dims):
    with h5py.File(file_path, 'r') as f:
        # shape(frame, neuron_num), temporal fluorescence traces
        if 'C' in f.keys():
            C = np.array(f.get('C')).T[:, :end_frame]
            neuron_num = C.shape[0]
        else:
            C = None

        if 'C_raw' in f.keys():
            C_raw = np.array(f.get('C_raw')).T[:, :end_frame]
            neuron_num = C_raw.shape[0]
        else:
            C_raw = None

        # C = np.array(f.get('C_raw')).T[:, :end_frame]
        # C = np.where(C < 0, 0, C)
        # neuron_num = C.shape[0]

        # shape(neuron_num, height * width), spatial components
        A = np.array(f.get('A'), dtype='float32').T.reshape(dims[0], dims[1], neuron_num)
        A = csc_matrix(A.transpose(1, 0, 2).reshape(-1, neuron_num))

    estimates = Estimates(A=A, b=None, C=C, f=None, R=None, dims=dims)
    estimates.C_raw = C_raw
    return estimates


def load_params(path):
    with open(path) as f:
        params  = json.loads(f.read())
    for category in params.keys():
        for key in params[category].keys():
            if type(params[category][key]) == list:
                params[category][key] = tuple(params[category][key])
    return params


if __name__ == "__main__":
    out_path, dims = ready_mov_file('/Users/hassaku/Documents/research/online-calcium-imaging/data/raw/20191017_130713/20191017_130713.mat', 'video', 100)
    estimates = load_annotation_file('data/raw/20191017_130713/20191017_130713.mat', 100, dims)
    print(estimates)

    out_path, dims = ready_mov_file('data/raw/DM95/DM95.h5', 'Object', 100)
    estimates = load_annotation_file('data/raw/DM95/DM95.mat', 100, dims)
    print(estimates)

    out_path, dims = ready_mov_file('data/raw/DM298/DM298_first10000.h5', 'mov', 100)
    estimates = load_annotation_file('data/raw/DM298/DM298.mat', 100, dims)
    print(estimates)
