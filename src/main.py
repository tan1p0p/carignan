import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    format="%(relativeCreated)12d [%(filename)s:%(funcName)10s():%(lineno)s] [%(process)d] %(message)s",
    level=logging.DEBUG
)

root = '../'
file_name = 'data/raw/20191017_130713/20191017_130713'
mov_ext = '.mat'
gt_ext = '.mat'
mov_key = 'video'
sys.path.append(root)

from src.data_handling import ready_mov_file, load_params

# out_paths = [ready_mov_file(root + file_name + mov_ext, mov_key, end_frame=200)[0]]
# for i in range(300):
#     out_path, dims = ready_mov_file(root + file_name + mov_ext, mov_key, first_frame=200+i, end_frame=200+i+1)
#     out_paths.append(out_path)

stride = 8                                      # overlap between patches (used only during initialization)
ssub = 2                                        # spatial downsampling factor (during initialization)
ds_factor = 1 * ssub                            # spatial downsampling factor (during online processing)
gSig = 8 // ds_factor
gSiz = 4 * gSig + 1
params_dict = {
    'fr': 5,
    'decay_time': 2,
    'gnb': 0,
    'epochs': 1,                                # number of passes over the data
    'nb': 0,                                    # need to set 0 for bare initialization
    'ssub': ssub,
    'ssub_B': 4,                                # background downsampling factor (use that for faster processing)
    'ds_factor': ds_factor,                     # ds_factor >= ssub should hold
    'gSig': (gSig, gSig),                       # expected half size of neurons
    'gSiz': (gSiz, gSiz),
    'gSig_filt': (3, 3),
    'max_shifts': (20, 20),                     # maximum allowed rigid shift
    'min_pnr': 3,                               # minimum peak-to-noise ratio for selecting a seed pixel.
    'min_corr': 0.3,                            # minimum local correlation coefficients for selecting a seed pixel.
    'bas_nonneg': False, 
    'center_psf': True,
    'max_shifts_online': 20,
    'rval_thr': 0.85,                            # correlation threshold for new component inclusion
    'motion_correct': True,
    'init_batch': 100,                          # number of frames for initialization (presumably from the first file)
    'only_init': True,
    
    'init_method': 'bare',
    # 'init_method': 'seeded',
    'n_pixels_per_process': 128,
    'normalize_init': False,
    'update_freq': 200,
    'expected_comps': 500,                       # maximum number of expected components used for memory pre-allocation (exaggerate here)
    'sniper_mode': False,                        # flag using a CNN to detect new neurons (o/w space correlation is used). set to False for 1p data       
    'dist_shape_update' : False,                 # flag for updating shapes in a distributed way
    'min_num_trial': 5,                          # number of candidate components per frame
    'use_corr_img': True,                        # flag for using the corr*pnr image when searching for new neurons (otherwise residual)
    'show_movie': False,                         # show the movie with the results as the data gets processed
    'motion_correct': True,
    'pw_rigid': False,                           # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    # 'full_XXt':True,
    
    'p': 1,                                      # order of AR indicator dynamics
    'center_psf': True,
    'merge_thr': 0.65,                           # merging threshold
    # 'min_SNR': 1.5,                              # minimum SNR for accepting new components
    
    'rf': 48,                                    # half size of patch (used only during initialization)
    'stride': 8,                                 # amount of overlap between patches   
    'border_nan': 'copy',                        # replicate values along the boundaries
    
    'K': None,
    
    # for CNMF-E.
    # From https://caiman.readthedocs.io/en/master/CaImAn_Tips.html?highlight=CNMF-E#p-processing-tips
    'center_psf': True,
    'method_init': 'corr_pnr',
    'ring_size_factor': 1.5,
    'only_init_patch': True,
    
    # 'simultaneously': True
}

from caiman.source_extraction import cnmf
from OnACID.real_time_cnmf import MiniscopeOnACID

opts = cnmf.params.CNMFParams(params_dict=params_dict)
cnm = MiniscopeOnACID(params=opts)
# cnm = cnmf.online_cnmf.OnACID(params=opts)
cnm.fit_from_scope(
    os.path.join(root, 'data/out/sample/sample'),
    # input_avi_path=os.path.join(root, 'data/interim/20191017_130713/20191017_130713.avi')
    input_avi_path=os.path.join(root, 'data/raw/LIS68HC/1msCam1HC.avi'),
    seed_file=os.path.join(root, 'data/interim/LIS68HC/seed.mat'),
)
