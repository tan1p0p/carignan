import datetime
import os
import sys

from caiman.source_extraction.cnmf import params

from calcium_imaging.CNMF import MiniscopeOnACID
from utils.utils import show_logs

root = './'

def prepare_onacid():
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

    opts = params.CNMFParams(params_dict=params_dict)
    onacid = MiniscopeOnACID(params=opts)
    return onacid

def run_onacid_from_file(cnm):
    cnm.fit_from_scope(
        out_file_name=os.path.join(root, 'data/out/sample/sample'),
        input_avi_path=os.path.join(root, 'data/raw/LIS68HC/1msCam1HC.avi'),
    )

def run_onacid_from_scope(cnm):
    now = str(datetime.datetime.now())
    out_dir = os.path.join(root, f'data/out/{now}/')
    os.makedirs(out_dir)
    cnm.fit_from_scope(
        input_camera_id=0,
        out_file_name=out_dir + 'out',
    )

if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1] in ['-s', '-f']:
        raise ValueError('please give me -s or -f option.')

    show_logs()
    cnm = prepare_onacid()
    if sys.argv[1] == '-s':
        run_onacid_from_scope(cnm)
    else:
        run_onacid_from_file(cnm)
