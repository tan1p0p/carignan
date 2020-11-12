import logging
import os
import multiprocessing
import sys
import time
from math import sqrt

import cv2
import h5py
import numpy as np
import caiman
from caiman.motion_correction import high_pass_filter_space, motion_correct_iteration_fast, sliding_window, tile_and_correct
from caiman.source_extraction.cnmf import online_cnmf, pre_processing, initialization
from cnmfereview.utils import crop_footprint, process_traces
from joblib import load
from scipy.sparse import csc_matrix
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

from modules.laser_handler import select_port, show_connection, shoot_laser

class MiniscopeOnACID(online_cnmf.OnACID):
    def __init__(self, params=None, estimates=None, path=None, dview=None):
        super().__init__(params=params, estimates=estimates, path=path, dview=dview)
        self.time_frame = 0
        self.seed_file = None
        self.sync_patterns = None
        self.checked_comps = 0

    def __init_window_status(self, frame_shape, max_bright):
        self.window_name = 'microscope CNMF-E'

        # seekbar texts
        self.gain_text = 'gain'
        self.fps_text = '0: 05fps\n1: 10fps\n2: 15fps\n3: 30fps\n4: 60fps'
        self.x0_text, self.x1_text = 'set_x0', 'set_x1'
        self.y0_text, self.y1_text = 'set_y0', 'set_y1'
        self.dr_max_text = 'dynamic range: max'
        self.dr_min_text = 'dynamic range: min'
        self.start_text = 'start analysis!'
        self.demixed_bias_text = 'demixed color bias'

        # params
        h, w = frame_shape
        self.fps = 30
        self.demixed_bias = 1
        self.x0 = 0
        self.x1 = w
        self.y0 = 0
        self.y1 = h
        self.dr_min = 0
        self.dr_max = max_bright
        self.is_shoot = False

    def __init_serial_status(self):
        self.serial_power = 60000
        self.serial_seconds = 0.5
        self.ser = select_port()
        show_connection(self.ser)

    def __init_models(self):
        self.model_dir = './models'
        # self.askl_model = load(os.path.join(self.model_dir, 'cr_tutorial_askl.joblib'))
        self.tpot_model = load(os.path.join(self.model_dir, 'cr_tutorial_tpot.joblib'))

    def __set_gain(self, x):
        gain = [16, 32, 64]
        cap.set(14, gain[x])
        time.sleep(0.01)
        logging.info(f'camera gain was set to {self.cap.get(14)}')

    def __set_fps(self, x):
        self.fps = [5, 10, 15, 30, 60][x]
        logging.info(f'fps was set to {self.fps}')

    def __set_plot(self, t, frame_shape, with_demixed=True):
        h, w = frame_shape
        half_w, half_h = w//2, h//2
        even_w, even_h = half_w*2, half_h*2

        bg = self.estimates.b0.reshape(self.estimates.dims, order='f').astype('uint8')
        if bg.shape[:2] != (half_h, half_w):
            bg = cv2.resize(bg, (half_w, half_h), interpolation=cv2.INTER_AREA)

        frame_cor = self.current_frame_cor.astype('uint8')
        if frame_cor.shape[:2] != (half_h, half_w):
            frame_cor = cv2.resize(self.current_frame_cor, (half_w, half_h), interpolation=cv2.INTER_AREA)

        diff = frame_cor - bg
        diff[frame_cor < bg] = 0

        plots = np.zeros((h, w), dtype='uint8')
        plots[:half_h, :half_w] = frame_cor
        plots[:half_h, half_w:even_w] = bg
        plots[half_h:even_h, :half_w] = diff
        plots_mapped = cv2.applyColorMap(plots, cv2.COLORMAP_VIRIDIS)
        plots_mapped[half_h:, half_w:] = 0

        if with_demixed:
            A_f = self.estimates.Ab[:, self.params.get('init', 'nb'):] # (size, N)
            C_f = self.estimates.C_on[self.params.get('init', 'nb'):self.M, t-1] # (N)
            color_map = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]]
            colored = np.zeros((self.estimates.dims[0], self.estimates.dims[1], 3), dtype='uint8')
            for i, c in enumerate(color_map):
                tmp = A_f[:, i::len(color_map)] * C_f[i::len(color_map)] * self.demixed_bias
                tmp = tmp.astype('uint8')
                tmp = tmp.reshape((self.estimates.dims[0], self.estimates.dims[1]), order='f')
                for j in range(3):
                    if c[j] == 1:
                        colored[:, :, j] += tmp
            if colored.shape[:2] != (half_h, half_w):
                colored = cv2.resize(colored, (half_w, half_h), interpolation=cv2.INTER_AREA)
            plots_mapped[half_h:even_h, half_w:even_w] = colored

        self.plot = plots_mapped

    def __set_results(self, t):
        epochs = self.params.get('online', 'epochs')
        if self.params.get('online', 'normalize'):
            self.estimates.Ab = csc_matrix(self.estimates.Ab.multiply(
                self.img_norm.reshape(-1, order='F')[:, np.newaxis]))
        self.estimates.A, self.estimates.b = self.estimates.Ab[:, self.params.get('init', 'nb'):], self.estimates.Ab[:, :self.params.get('init', 'nb')].toarray()
        self.estimates.C, self.estimates.f = self.estimates.C_on[self.params.get('init', 'nb'):self.M, t - t //
                         epochs:t], self.estimates.C_on[:self.params.get('init', 'nb'), t - t // epochs:t]
        noisyC = self.estimates.noisyC[self.params.get('init', 'nb'):self.M, t - t // epochs:t]
        self.estimates.YrA = noisyC - self.estimates.C
        if self.estimates.OASISinstances is not None:
            self.estimates.bl = [osi.b for osi in self.estimates.OASISinstances]
            self.estimates.S = np.stack([osi.s for osi in self.estimates.OASISinstances])
            self.estimates.S = self.estimates.S[:, t - t // epochs:t]
        else:
            self.estimates.bl = [0] * self.estimates.C.shape[0]
            self.estimates.S = np.zeros_like(self.estimates.C)

    def __save_results(self, frame):
        if self.params.get('online', 'ds_factor') > 1:
            neuron_num = self.estimates.A.shape[-1]
            A = np.hstack([cv2.resize(self.estimates.A[:, i].reshape(self.estimates.dims, order='F').toarray(),
                                    frame.shape[::-1]).reshape(-1, order='F')[:,None] for i in range(neuron_num)])
        with h5py.File(self.out_mat_file, 'a') as f:
            f['A'].resize(A.shape)
            f['A'][()] = A
            f['C'].resize(self.estimates.C.shape)
            f['C'][()] = self.estimates.C
            f['S'].resize(self.estimates.S.shape)
            f['S'][()] = self.estimates.S

    def __prepare_window(self, mode, frame_shape, max_bright):
        h, w = frame_shape
        cv2.destroyAllWindows()
        cv2.namedWindow(self.window_name)
        if mode == 'prepare':
            cv2.createTrackbar(self.gain_text, self.window_name, 0, 2, self.__set_gain)
            cv2.createTrackbar(self.fps_text, self.window_name, 1, 4, self.__set_fps)
            cv2.createTrackbar(self.x0_text, self.window_name, 0, w, lambda x:x)
            cv2.createTrackbar(self.x1_text, self.window_name, w, w, lambda x:x)
            cv2.createTrackbar(self.y0_text, self.window_name, 0, h, lambda x:x)
            cv2.createTrackbar(self.y1_text, self.window_name, h, h, lambda x:x)
            cv2.createTrackbar(self.start_text, self.window_name, 0, 1, lambda x: True if x == 0 else False)
        elif mode == 'analyze':
            cv2.createTrackbar(self.demixed_bias_text, self.window_name, 1, 10, lambda x:x)
        
        if mode != 'initialize':
            cv2.createTrackbar(self.dr_min_text, self.window_name, 0, max_bright, lambda x:x)
            cv2.createTrackbar(self.dr_max_text, self.window_name, max_bright, max_bright, lambda x:x)

    def __show_next_frame(self, text, mode, text_color=(255, 255, 255), avi_out=None):
        _, frame = self.cap.read()

        if mode == 'prepare':
            cv2.getTrackbarPos(self.gain_text, self.window_name)
            self.x0 = cv2.getTrackbarPos(self.x0_text, self.window_name)
            self.x1 = cv2.getTrackbarPos(self.x1_text, self.window_name)
            self.y0 = cv2.getTrackbarPos(self.y0_text, self.window_name)
            self.y1 = cv2.getTrackbarPos(self.y1_text, self.window_name)
            cv2.getTrackbarPos(self.fps_text, self.window_name)
        elif mode == 'analyze':
            self.demixed_bias = cv2.getTrackbarPos(self.demixed_bias_text, self.window_name)

        frame = frame[self.y0:self.y1, self.x0:self.x1]
        if avi_out != None:
            avi_out.write(frame)
        out_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if mode != 'initialize':
            self.dr_min = cv2.getTrackbarPos(self.dr_min_text, self.window_name)
            self.dr_max = cv2.getTrackbarPos(self.dr_max_text, self.window_name)
            frame = frame.astype('float')
            frame[frame < self.dr_min] = self.dr_min
            frame[frame > self.dr_max] = self.dr_max
            frame -= self.dr_min
            frame *= 255 / (self.dr_max - self.dr_min)
            frame = frame.astype('uint8')

        if mode == 'analyze':
            frame = np.hstack([frame, self.plot])

        cv2.putText(frame, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color)
        cv2.imshow(self.window_name, frame)
        return out_frame

    # TODO: need to refactor
    def __get_model_LN(self):
        if self.params.get('online', 'ring_CNN'):
            logging.info('Using Ring CNN model')
            from caiman.utils.nn_models import (fit_NL_model, create_LN_model, quantile_loss, rate_scheduler)
            gSig = self.params.get('init', 'gSig')[0]
            width = self.params.get('ring_CNN', 'width')
            nch = self.params.get('ring_CNN', 'n_channels')
            if self.params.get('ring_CNN', 'loss_fn') == 'pct':
                loss_fn = quantile_loss(self.params.get('ring_CNN', 'pct'))
            else:
                loss_fn = self.params.get('ring_CNN', 'loss_fn')
            if self.params.get('ring_CNN', 'lr_scheduler') is None:
                sch = None
            else:
                sch = rate_scheduler(*self.params.get('ring_CNN', 'lr_scheduler'))
            Y = caiman.base.movies.load(fls[0],
                                        subindices=slice(self.params.get('online', 'init_batch')),
                                        var_name_hdf5=self.params.get('data', 'var_name_hdf5'))
            shape = Y.shape[1:] + (1,)
            logging.info('Starting background model training.')
            model_LN = create_LN_model(Y, shape=shape, n_channels=nch,
                                       lr=self.params.get('ring_CNN', 'lr'), gSig=gSig,
                                       loss=loss_fn, width=width,
                                       use_add=self.params.get('ring_CNN', 'use_add'),
                                       use_bias=self.params.get('ring_CNN', 'use_bias'))
            if self.params.get('ring_CNN', 'reuse_model'):
                logging.info('Using existing model from {}'.format(self.params.get('ring_CNN', 'path_to_model')))
                model_LN.load_weights(self.params.get('ring_CNN', 'path_to_model'))
            else:
                logging.info('Estimating model from scratch, starting training.')
                model_LN, history, path_to_model = fit_NL_model(model_LN, Y,
                                                                epochs=self.params.get('ring_CNN', 'max_epochs'),
                                                                patience=self.params.get('ring_CNN', 'patience'),
                                                                schedule=sch)
                logging.info('Training complete. Model saved in {}.'.format(path_to_model))
                self.params.set('ring_CNN', {'path_to_model': path_to_model})
        else:
            model_LN = None
        return model_LN

    # TODO: need to refactor
    def __initialize_online(self, model_LN, Y):
        _, original_d1, original_d2 = Y.shape
        opts = self.params.get_group('online')
        init_batch = opts['init_batch']
        if model_LN is not None:
            Y = Y - caiman.movie(np.squeeze(model_LN.predict(np.expand_dims(Y, -1))))
            Y = np.maximum(Y, 0)
        # Downsample if needed
        ds_factor = np.maximum(opts['ds_factor'], 1)
        if ds_factor > 1:
            Y = Y.resize(1./ds_factor, 1./ds_factor)
        self.estimates.shifts = []  # store motion shifts here
        self.estimates.time_new_comp = []
        if self.params.get('online', 'motion_correct'):
            max_shifts_online = self.params.get('online', 'max_shifts_online')
            if self.params.get('motion', 'gSig_filt') is None:
                mc = Y.motion_correct(max_shifts_online, max_shifts_online)
                Y = mc[0].astype(np.float32)
            else:
                Y_filt = np.stack([high_pass_filter_space(yf, self.params.motion['gSig_filt']) for yf in Y], axis=0)
                Y_filt = caiman.movie(Y_filt)
                mc = Y_filt.motion_correct(max_shifts_online, max_shifts_online)
                Y = Y.apply_shifts(mc[1])
            if self.params.get('motion', 'pw_rigid'):
                n_p = len([(it[0], it[1])
                     for it in sliding_window(Y[0], self.params.get('motion', 'overlaps'), self.params.get('motion', 'strides'))])
                for sh in mc[1]:
                    self.estimates.shifts.append([tuple(sh) for i in range(n_p)])
            else:
                self.estimates.shifts.extend(mc[1])                
        self.img_min = Y.min()
        self.current_frame_cor = Y[-1]

        if self.params.get('online', 'normalize'):
            Y -= self.img_min
        img_norm = np.std(Y, axis=0)
        img_norm += np.median(img_norm)  # normalize data to equalize the FOV
        logging.info('Frame size:' + str(img_norm.shape))
        if self.params.get('online', 'normalize'):
            Y = Y/img_norm[None, :, :]
        total_frame, d1, d2 = Y.shape
        Yr = Y.to_2D().T        # convert data into 2D array
        self.img_norm = img_norm
        if self.params.get('online', 'init_method') == 'bare':
            logging.info('Using bare init')
            init = self.params.get_group('init').copy()
            is1p = (init['method_init'] == 'corr_pnr' and  init['ring_size_factor'] is not None)
            if is1p:
                self.estimates.sn, psx = pre_processing.get_noise_fft(
                    Yr, noise_range=self.params.get('preprocess', 'noise_range'),
                    noise_method=self.params.get('preprocess', 'noise_method'),
                    max_num_samples_fft=self.params.get('preprocess', 'max_num_samples_fft'))
            for key in ('K', 'nb', 'gSig', 'method_init'):
                init.pop(key, None)
            tmp = online_cnmf.bare_initialization(
                Y.transpose(1, 2, 0), init_batch=self.params.get('online', 'init_batch'),
                k=self.params.get('init', 'K'), gnb=self.params.get('init', 'nb'),
                method_init=self.params.get('init', 'method_init'), sn=self.estimates.sn,
                gSig=self.params.get('init', 'gSig'), return_object=False,
                options_total=self.params.to_dict(), **init)
            if is1p:
                (self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f,
                 self.estimates.YrA, self.estimates.W, self.estimates.b0) = tmp
            else:
                (self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f,
                 self.estimates.YrA) = tmp
            self.__init_models()
            self.__reject_fp_comps()

            self.estimates.S = np.zeros_like(self.estimates.C)
            nr = self.estimates.C.shape[0]
            self.estimates.g = np.array([-np.poly([0.9] * max(self.params.get('preprocess', 'p'), 1))[1:]
                               for gg in np.ones(nr)])
            self.estimates.bl = np.zeros(nr)
            self.estimates.c1 = np.zeros(nr)
            self.estimates.neurons_sn = np.std(self.estimates.YrA, axis=-1)
            self.estimates.lam = np.zeros(nr)
        elif self.params.get('online', 'init_method') == 'cnmf':
            n_processes = cpu_count() - 1 or 1
            cnm = CNMF(n_processes=n_processes, params=self.params, dview=self.dview)
            cnm.estimates.shifts = self.estimates.shifts
            if self.params.get('patch', 'rf') is None:
                cnm.dview = None
                cnm.fit(np.array(Y))
                self.estimates = cnm.estimates

            else:
                Y.save('init_file.hdf5')
                f_new = mmapping.save_memmap(['init_file.hdf5'], base_name='Yr', order='C',
                                             slices=[slice(0, opts['init_batch']), None, None])

                Yrm, dims_, T_ = mmapping.load_memmap(f_new)
                Y = np.reshape(Yrm.T, [T_] + list(dims_), order='F')
                cnm.fit(Y)
                self.estimates = cnm.estimates
                if self.params.get('online', 'normalize'):
                    self.estimates.A /= self.img_norm.reshape(-1, order='F')[:, np.newaxis]
                    self.estimates.b /= self.img_norm.reshape(-1, order='F')[:, np.newaxis]
                    self.estimates.A = csc_matrix(self.estimates.A)
        elif self.params.get('online', 'init_method') == 'seeded':
            init = self.params.get_group('init').copy()
            is1p = (init['method_init'] == 'corr_pnr' and init['ring_size_factor'] is not None)

            if self.seed_file is None:
                raise ValueError('Please input analyzed mat file path as seed_file.')
            with h5py.File(self.seed_file, 'r') as f:
                ds_factor = self.params.get('online', 'ds_factor')
                Ain = f['A'][()].reshape((original_d1, original_d2, -1), order='F')
                Ain = cv2.resize(Ain, (d2, d1))
                Ain = Ain.reshape((-1, Ain.shape[-1]), order='F')
                Ain_norm = (Ain - Ain.min(0)[None, :]) / (Ain.max(0) - Ain.min(0))
                A_seed = Ain_norm > 0.5

            tmp = online_cnmf.seeded_initialization(
                Y.transpose(1, 2, 0), A_seed, k=self.params.get('init', 'K'),
                gSig=self.params.get('init', 'gSig'), return_object=False)
            self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f, self.estimates.YrA = tmp

            if is1p:
                ssub_B = self.params.get('init', 'ssub_B') * self.params.get('init', 'ssub')
                ring_size_factor = self.params.get('init', 'ring_size_factor')
                gSiz = 2 * np.array(self.params.get('init', 'gSiz')) // 2 + 1
                W, b0 = initialization.compute_W(
                    Y.transpose(1, 2, 0).reshape((-1, total_frame), order='F'),
                    self.estimates.A, self.estimates.C, (d1, d2), ring_size_factor * gSiz[0], ssub=ssub_B)
                self.estimates.W, self.estimates.b0 = W, b0
            self.estimates.S = np.zeros_like(self.estimates.C)
            nr = self.estimates.C.shape[0]
            self.estimates.g = np.array([-np.poly([0.9] * max(self.params.get('preprocess', 'p'), 1))[1:]
                               for gg in np.ones(nr)])
            self.estimates.bl = np.zeros(nr)
            self.estimates.c1 = np.zeros(nr)
            self.estimates.neurons_sn = np.std(self.estimates.YrA, axis=-1)
            self.estimates.lam = np.zeros(nr)
        else:
            raise Exception('Unknown initialization method!')
        T1 = init_batch * self.params.get('online', 'epochs')
        self.params.set('data', {'dims': Y.shape[1:]})
        self._prepare_object(Yr, T1)
        return self

    def __fit_next_from_raw(self, frame, t, model_LN=None, out=None):
        ssub_B = self.params.get('init', 'ssub_B') * self.params.get('init', 'ssub')
        d1, d2 = self.params.get('data', 'dims')
        max_shifts_online = self.params.get('online', 'max_shifts_online')

        if model_LN is not None:
            if self.params.get('ring_CNN', 'remove_activity'):
                activity = self.estimates.Ab[:,:self.N].dot(self.estimates.C_on[:self.N, t-1]).reshape(self.params.get('data', 'dims'), order='F')
                if self.params.get('online', 'normalize'):
                    activity *= self.img_norm
            else:
                activity = 0.
                # frame = frame.astype(np.float32) - activity
            frame = frame - np.squeeze(model_LN.predict(np.expand_dims(np.expand_dims(frame.astype(np.float32) - activity, 0), -1)))
            frame = np.maximum(frame, 0)

        t_frame_start = time.time()
        if np.isnan(np.sum(frame)):
            raise Exception('Current frame contains NaN')

        frame_ = frame.copy().astype(np.float32)
        if self.params.get('online', 'ds_factor') > 1:
            frame_ = cv2.resize(frame_, self.img_norm.shape[::-1])

        if self.params.get('online', 'normalize'):
            frame_ -= self.img_min     # make data non-negative

        if self.params.get('online', 'motion_correct'):
            templ = self.estimates.Ab.dot(
                    np.median(self.estimates.C_on[:self.M, t-51:t-1], 1)).reshape(self.params.get('data', 'dims'), order='F')#*self.img_norm
            if self.is1p and self.estimates.W is not None:
                if ssub_B == 1:
                    B = self.estimates.W.dot((frame_ - templ).flatten(order='F') - self.estimates.b0) + self.estimates.b0
                    B = B.reshape(self.params.get('data', 'dims'), order='F')
                else:
                    b0 = self.estimates.b0.reshape((d1, d2), order='F')#*self.img_norm
                    bc2 = initialization.downscale(frame_ - templ - b0, (ssub_B, ssub_B)).flatten(order='F')
                    Wb = self.estimates.W.dot(bc2).reshape(((d1 - 1) // ssub_B + 1, (d2 - 1) // ssub_B + 1), order='F')
                    B = b0 + np.repeat(np.repeat(Wb, ssub_B, 0), ssub_B, 1)[:d1, :d2]
                templ += B
            if self.params.get('online', 'normalize'):
                templ *= self.img_norm
            if self.is1p:
                templ = high_pass_filter_space(templ, self.params.motion['gSig_filt'])
            if self.params.get('motion', 'pw_rigid'):
                frame_cor, shift, _, xy_grid = tile_and_correct(frame_, templ, self.params.motion['strides'], self.params.motion['overlaps'],
                                                                self.params.motion['max_shifts'], newoverlaps=None, newstrides=None, upsample_factor_grid=4,
                                                                upsample_factor_fft=10, show_movie=False, max_deviation_rigid=self.params.motion['max_deviation_rigid'],
                                                                add_to_movie=0, shifts_opencv=True, gSig_filt=None,
                                                                use_cuda=False, border_nan='copy')
            else:
                if self.is1p:
                    frame_orig = frame_.copy()
                    frame_ = high_pass_filter_space(frame_, self.params.motion['gSig_filt'])
                frame_cor, shift = motion_correct_iteration_fast(
                        frame_, templ, max_shifts_online, max_shifts_online)
                if self.is1p:
                    M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
                    frame_cor = cv2.warpAffine(
                        frame_orig, M, frame_.shape[::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

            self.estimates.shifts.append(shift)
        else:
            templ = None
            frame_cor = frame_

        self.current_frame_cor = frame_cor

        if self.params.get('online', 'normalize'):
            frame_cor = frame_cor/self.img_norm
        self.fit_next(t, frame_cor.reshape(-1, order='F'))

    def __reject_fp_comps(self):
        trace_len = 500
        if self.estimates.C.shape[1] < trace_len:
            return False

        spatial = np.asarray(self.estimates.A[:, self.checked_comps:])
        spatial = spatial.reshape(self.estimates.dims + (-1,), order='F').transpose(2, 0, 1)
        spatial = crop_footprint(spatial, 80)
        spatial = spatial.reshape((spatial.shape[0], -1))
        trace = process_traces(self.estimates.C[self.checked_comps:], trace_len)
        combined = np.concatenate((spatial, trace), axis=1)
        breakpoint()

        model = 'tpot'
        if model == 'askl':
            pred = self.askl_model.predict(combined)
        elif model == 'tpot':
            pred = self.tpot_model.predict(combined)
        elif model == 'deep':
            pred = self.deep_model.predict(combined)
        else:
            raise ValueError('Unsupported model!!')

        pred < 0.5
        breakpoint()

        pass # reject fp comps
        self.checked_comps += 100 # update checked_comps

    def __shoot_laser(self):
        shoot_laser(self.ser, self.serial_power, self.serial_seconds)

    def fit_from_scope(self, out_file_name, input_camera_id=0, input_avi_path=None,
                       seed_file=None, sync_pattern_file=None, **kargs):
        self.seed_file = seed_file
        self.out_mat_file = out_file_name + '.mat'
        self.out_avi_file = out_file_name + '.avi'
        dir_path = os.path.dirname(out_file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if sync_pattern_file != None:
            with h5py.File(sync_pattern_file, 'r') as f:
                self.sync_patterns = f['sync_pattern'][()]
            self.__init_serial_status()

        # set some camera params
        if input_avi_path is None:
            self.cap = cv2.VideoCapture(input_camera_id)
        else:
            self.cap = cv2.VideoCapture(input_avi_path)

        model_LN = self.__get_model_LN()
        ret, frame = self.cap.read()
        if not ret:
            raise Exception('frame cannot read.')
        max_h, max_w, _ = frame.shape
        max_bright = max(255, frame.max())
        self.__init_window_status((max_h, max_w), max_bright)
        self.__prepare_window(mode='prepare', frame_shape=(max_h, max_w), max_bright=max_bright)

        prev_time = time.time()
        while True:
            time_d = 1 / self.fps
            while time.time() - prev_time < time_d:
                pass
            prev_time = time.time()
            frame = self.__show_next_frame('prepareing...', mode='prepare')
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getTrackbarPos(self.start_text, self.window_name):
                break

        h, w = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        avi_out = cv2.VideoWriter(self.out_avi_file, fourcc, 10.0, (w, h))

        self.__prepare_window(mode='initialize', frame_shape=(h, w), max_bright=max_bright)
        init_Y = np.empty((self.params.get('online', 'init_batch'),) + frame.shape)

        with h5py.File(self.out_mat_file, 'w') as f:
            f['initialize_first_frame_t'] = time.time()

        prev_time = time.time()
        time_d = 1 / self.fps
        for i in range(self.params.get('online', 'init_batch')):
            while time.time() - prev_time < time_d:
                pass
            prev_time = time.time()
            frame = self.__show_next_frame('initialize', mode='initialize', avi_out=avi_out)
            init_Y[i] = frame.copy()
            cv2.waitKey(1)
        self.time_frame += self.params.get('online', 'init_batch')

        with h5py.File(self.out_mat_file, 'a') as f:
            f['initialize_last_frame_t'] = time.time()

        Y_init = caiman.base.movies.movie(init_Y.astype(np.float32))
        self.__initialize_online(model_LN=model_LN, Y=Y_init)
        self.__prepare_window(mode='analyze', frame_shape=(h, w), max_bright=max_bright)

        with h5py.File(self.out_mat_file, 'a') as f:
            f['cnmfe_first_frame_t'] = time.time()
            A_size = frame.shape[0] * frame.shape[1]
            f.create_dataset('A', (A_size, self.N), maxshape=(A_size, None))
            f.create_dataset('C', (self.N, 100), maxshape=(None, None))
            f.create_dataset('S', (self.N, 100), maxshape=(None, None))

        prev_time = time.time()
        while True:
            while time.time() - prev_time < time_d:
                pass
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()

            # try:
            self.__set_plot(self.time_frame, frame_shape=(h, w), with_demixed=True)
            if self.time_frame % 100 == 0:
                self.__set_results(self.time_frame)
                self.estimates.noisyC = np.hstack(
                    (self.estimates.noisyC, np.zeros((self.estimates.noisyC.shape[0], 100))))
                self.estimates.C_on = np.hstack(
                    (self.estimates.C_on, np.zeros((self.estimates.C_on.shape[0], 100))))
                p = multiprocessing.Process(target=self.__save_results, args=[frame])
                p.start()

            comp_num = self.M - self.params.get('init', 'nb')
            if self.is_shoot:
                frame = self.__show_next_frame(f'FPS: {fps:.4f}, neurons: {comp_num}, SHOOT!!!', mode='analyze', avi_out=avi_out, text_color=(255, 0, 0))
            else:
                frame = self.__show_next_frame(f'FPS: {fps:.4f}, neurons: {comp_num}', mode='analyze', avi_out=avi_out)

            self.__fit_next_from_raw(frame, self.time_frame, model_LN=model_LN)
            if self.sync_patterns:
                latest = self.estimates.C_on[self.params.get('init', 'nb'):self.M, t-1:t].squeeze()
                self.is_shoot = False
                if np.any(np.all(self.sync_patterns < latest, axis=1)):
                    self.__shoot_laser()
                    self.is_shoot = True

            self.time_frame += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # except:
            #     print(sys.exc_info())
        
        with h5py.File(self.out_mat_file, 'a') as f:
            f['cnmfe_last_frame_t'] = time.time()
            f.create_dataset('b0', data=self.estimates.b0)
            f.create_dataset('W', data=self.estimates.W.toarray())
        avi_out.release()
        try:
            self.ser.close()
        except:
            pass
        return self
