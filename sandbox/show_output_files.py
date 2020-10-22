import h5py

with h5py.File('data/out/sample/sample.mat','r') as f:
    for key in f.keys():
        print(f[key])
