import cv2
import h5py
import numpy as np
from scipy import interpolate

with h5py.File('data/interim/DM108/DM108_video.h5', 'r') as f:
    full_movie = f['Object'][()]

# with h5py.File('data/interim/DM108/DM108_video.h5', 'w') as f:
#     f.create_dataset('Object', data=movie)

for num, frame_count in enumerate([(0, 3000), (3000, 6000), (6000, 7500)]):
    movie = full_movie[frame_count[0]:frame_count[1]]
    sample_pixels = sorted(np.random.choice(movie.flatten(), size=movie.shape[1]*movie.shape[2]))

    value_index = {}
    max_i = len(sample_pixels)
    for i, v in enumerate(sample_pixels):
        value_index[v] = i / max_i * 255

    cul_v = np.array(list(value_index.keys()))
    new_v = np.array(list(value_index.values()))
    f = interpolate.interp1d(cul_v, new_v, axis=0, fill_value="extrapolate")
    print('index prepared')

    cul_v_all = np.arange(movie.min(), movie.max()+1)
    new_v_all = f(cul_v_all)

    cul_new = {}
    for cul, new in zip(cul_v_all, new_v_all):
        cul_new[cul] = new
        
    mp = np.arange(0, movie.max()+1)
    mp[list(cul_new.keys())] = list(cul_new.values())
    movie = mp[movie].astype('uint8')

    print('start video writing')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        f'./data/interim/DM108/DM108_comp-{num}.avi',
        fourcc, 10.0, (movie.shape[2], movie.shape[1]))

    for frame_ in movie:
        frame = np.stack((frame_, frame_, frame_)).transpose(1,2,0)
        cv2.imshow('frame', frame)
        out.write(frame)
        cv2.waitKey(1)
    out.release()
