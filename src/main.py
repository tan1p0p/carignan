import datetime
import os
import sys

from caiman.source_extraction.cnmf import params

from modules.cnmf import MiniscopeOnACID
from modules.utils import show_logs


def get_caiman_params():
    stride = 8                                      # overlap between patches (used only during initialization)
    ssub = 2                                        # spatial downsampling factor (during initialization)
    ds_factor = 1 * ssub                            # spatial downsampling factor (during online processing)
    gSig = 8 // ds_factor
    gSiz = 4 * gSig + 1
    params_dict = {
        'fr': 10,
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
        'init_batch': 500,                         # number of frames for initialization (presumably from the first file)

        'init_method': 'bare',
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
        
        'p': 1,                                      # order of AR indicator dynamics
        'center_psf': True,
        'merge_thr': 0.65,                           # merging threshold
        # 'min_SNR': 1.5,                              # minimum SNR for accepting new components
        
        'rf': 48,                                    # half size of patch (used only during initialization)
        'stride': 8,                                 # amount of overlap between patches   
        'border_nan': 'copy',                        # replicate values along the boundaries
        
        'K': None,
        
        # From https://caiman.readthedocs.io/en/master/CaImAn_Tips.html?highlight=CNMF-E#p-processing-tips
        'center_psf': True,
        'method_init': 'corr_pnr',
        'ring_size_factor': 1.5,
        'only_init_patch': True,
    }

    return params.CNMFParams(params_dict=params_dict)

def run_onacid_from_file(caiman_params):
    method = 'CNN' # TPOT, CNN or ShufflenetV2
    onacid = MiniscopeOnACID(
        # seed_file='data/interim/DM108/DM108_A.mat',
        # sync_pattern_file='data/interim/DM108/DM108_sync-1.mat',
        fp_detect_method=method,
        caiman_params=caiman_params)
    onacid.fit_from_file(
        input_video_path='data/raw/neurofinder/neurofinder.01.00/images/',
        # input_video_path='data/raw/LIS68HC/1msCam1HC.avi',
        # input_video_path='data/raw/LIS68HC/1msCam1HC_small.avi',
        # input_video_path='data/interim/DM108/DM108_video.h5',
        mov_key='Object',
        output_dir=f'data/out/neurofinder.01.00/{method}/',
        with_plot=True
    )

def run_onacid_from_scope(caiman_params):
    onacid = MiniscopeOnACID(
        caiman_params=caiman_params)
    onacid.fit_from_scope(
        input_camera_id=0,
        output_dir=f'data/out/{datetime.datetime.now()}/',
    )

if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1] in ['-s', '-f']:
        raise ValueError('please give me -s or -f option.')

    show_logs()
    caiman_params = get_caiman_params()
    if sys.argv[1] == '-s':
        run_onacid_from_scope(caiman_params)
    else:
        run_onacid_from_file(caiman_params)
