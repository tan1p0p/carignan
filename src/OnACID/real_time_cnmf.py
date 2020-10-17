import logging
import os
import time

import cv2
import h5py
import numpy as np
import caiman
from caiman.motion_correction import high_pass_filter_space, motion_correct_iteration_fast, sliding_window, tile_and_correct
from caiman.source_extraction.cnmf import online_cnmf, pre_processing, initialization

class MiniscopeOnACID(online_cnmf.OnACID):
    def __init__(self, params=None, estimates=None, path=None, dview=None):
        super().__init__(params=params, estimates=estimates, path=path, dview=dview)

    def initialize_online(self, model_LN=None, fls=None, init_batch=None, Y=None):
        if fls == None:
            fls = self.params.get('data', 'fnames')
        opts = self.params.get_group('online')
        if init_batch == None:
            init_batch = opts['init_batch']
        if type(Y) != caiman.base.movies.movie:
            mode = 'from_file'
            Y = caiman.load(fls[0], subindices=slice(0, init_batch,
                        None), var_name_hdf5=self.params.get('data', 'var_name_hdf5')).astype(np.float32)
        else:
            mode = 'from_scope'
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
        img_min = Y.min()
        self.current_frame_cor = Y[-1]

        if self.params.get('online', 'normalize'):
            Y -= img_min
        img_norm = np.std(Y, axis=0)
        img_norm += np.median(img_norm)  # normalize data to equalize the FOV
        logging.info('Frame size:' + str(img_norm.shape))
        if self.params.get('online', 'normalize'):
            Y = Y/img_norm[None, :, :]
        if opts['show_movie']:
            self.bnd_Y = np.percentile(Y,(0.001,100-0.001))
        total_frame, d1, d2 = Y.shape
        Yr = Y.to_2D().T        # convert data into 2D array
        self.img_min = img_min
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
            tmp = seeded_initialization(
                Y.transpose(1, 2, 0), self.estimates.A, gnb=self.params.get('init', 'nb'), k=self.params.get('init', 'K'),
                gSig=self.params.get('init', 'gSig'), return_object=False)
            self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f, self.estimates.YrA = tmp
            if is1p:
                ssub_B = self.params.get('init', 'ssub_B') * self.params.get('init', 'ssub')
                ring_size_factor = self.params.get('init', 'ring_size_factor')
                gSiz = 2 * np.array(self.params.get('init', 'gSiz')) // 2 + 1
                W, b0 = compute_W(
                    Y.transpose(1, 2, 0).reshape((-1, total_frame), order='F'),
                    tmp[0], tmp[2], (d1, d2), ring_size_factor * gSiz[0], ssub=ssub_B)
                self.estimates.W, self.estimates.b0 = W, b0
            # <class 'scipy.sparse.csr.csr_matrix'> (1410, 1410) <class 'numpy.ndarray'> (22560,)
            # <class 'scipy.sparse.csr.csr_matrix'> (1410, 1410) <class 'numpy.ndarray'> (90240,)
            # print(type(self.estimates.W), self.estimates.W.shape, type(self.estimates.b0), self.estimates.b0.shape)
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
        if mode == 'from_scope':
            T1 = init_batch*self.params.get('online', 'epochs')
        else:
            _, Ts = get_file_size(fls, var_name_hdf5=self.params.get('data', 'var_name_hdf5'))
            T1 = np.array(Ts).sum()*self.params.get('online', 'epochs')
        dims = Y.shape[1:]
        self.params.set('data', {'dims': dims})
        logging.info('before prepare')
        self._prepare_object(Yr, T1)
        logging.info('after prepare')
        if opts['show_movie']:
            self.bnd_AC = np.percentile(self.estimates.A.dot(self.estimates.C),
                                        (0.001, 100-0.005))
            #self.bnd_BG = np.percentile(self.estimates.b.dot(self.estimates.f),
            #                            (0.001, 100-0.001))
        logging.info('end')
        return self

    def fit_next_from_raw(self, frame, t, model_LN=None, out=None):
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

        # Downsample and normalize
        frame_ = frame.copy().astype(np.float32)
        if self.params.get('online', 'ds_factor') > 1:
            frame_ = cv2.resize(frame_, self.img_norm.shape[::-1])

        if self.params.get('online', 'normalize'):
            frame_ -= self.img_min     # make data non-negative
        t_mot = time.time()

        # Motion Correction
        if self.params.get('online', 'motion_correct'):    # motion correct
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

        self.t_motion.append(time.time() - t_mot)
        self.current_frame_cor = frame_cor

        if self.params.get('online', 'normalize'):
            frame_cor = frame_cor/self.img_norm
        # Fit next frame
        self.fit_next(t, frame_cor.reshape(-1, order='F')) # MEMO: ここでfit_nextが呼ばれてる→その前に色々ある
        # Show
        if self.params.get('online', 'show_movie'):
            self.t = t
            vid_frame = self.create_frame(frame_cor, resize_fact=resize_fact)
            if self.params.get('online', 'save_online_movie'):
                out.write(vid_frame)
                for rp in range(len(self.estimates.ind_new)*2):
                    out.write(vid_frame)

            cv2.imshow('frame', vid_frame)
            for rp in range(len(self.estimates.ind_new)*2):
                cv2.imshow('frame', vid_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
                pass
        return time.time() - t_frame_start

    def fit_from_scope(self, out_file_name, input_avi_path=None, seed_file=None, **kargs):
        init_batch = self.params.get('online', 'init_batch')
        epochs = self.params.get('online', 'epochs')
        dir_path = os.path.dirname(out_file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # set some camera params
        if input_avi_path is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(input_avi_path)
        window_name = 'prev'

        # TODO: Need to fix here later
        model_LN = self.get_model_LN()

        # create switch for choose GAIN
        def set_GAIN(x):
            gain = [16, 32, 64]
            print(cap.set(14, gain[x]))
            time.sleep(0.01)
            print(f'gain: {cap.get(14)}')
        gain_text = 'gain'

        # create switch for choose FPS
        self.fps = 30
        def set_FPS(x):
            self.fps = [5, 10, 15, 30, 60][x]
            print(f'fps: {self.fps}')

        self.demixed_bias = 1
        def set_bias(x):
            self.demixed_bias = x

        fps_text = '0: 05fps\n1: 10fps\n2: 15fps\n3: 30fps\n4: 60fps'
        x0_text, x1_text = 'set_x0', 'set_x1'
        y0_text, y1_text = 'set_y0', 'set_y1'
        dr_max_text = 'dynamic range: max'
        dr_min_text = 'dynamic range: min'
        bias_text = 'demixed color bias'

        ret, frame = cap.read()
        if not ret:
            raise Exception('frame cannot read.')
        max_h, max_w, _ = frame.shape
        win_params = {}
        max_bright = max(255, frame.max())

        def prepare_window(win_params, mode):
            cv2.destroyAllWindows()
            cv2.namedWindow(window_name)
            cv2.createTrackbar(gain_text, window_name, 0, 2, set_GAIN)
            if mode == 'prepare':
                cv2.createTrackbar(fps_text, window_name, 1, 4, set_FPS)
                cv2.createTrackbar(x0_text, window_name, 0, max_w, lambda x:x)
                cv2.createTrackbar(x1_text, window_name, max_w, max_w, lambda x:x)
                cv2.createTrackbar(y0_text, window_name, 0, max_h, lambda x:x)
                cv2.createTrackbar(y1_text, window_name, max_h, max_h, lambda x:x)
            elif mode == 'analyze':
                cv2.createTrackbar(bias_text, window_name, 1, 10, set_bias)
            cv2.createTrackbar(dr_min_text, window_name, 0, max_bright, lambda x:x)
            cv2.createTrackbar(dr_max_text, window_name, max_bright, max_bright, lambda x:x)

        def show_next_frame(text, win_params, mode, text_color=(255, 255, 255), avi_out=None):
            _, frame = cap.read()
            if mode == 'prepare':
                win_params['x0'] = cv2.getTrackbarPos(x0_text, window_name)
                win_params['x1'] = cv2.getTrackbarPos(x1_text, window_name)
                win_params['y0'] = cv2.getTrackbarPos(y0_text, window_name)
                win_params['y1'] = cv2.getTrackbarPos(y1_text, window_name)
                cv2.getTrackbarPos(fps_text, window_name)
            elif mode == 'analyze':
                cv2.getTrackbarPos(bias_text, window_name)
            cv2.getTrackbarPos(gain_text, window_name)

            frame = frame[win_params['y0']:win_params['y1'], win_params['x0']:win_params['x1']]
            if avi_out != None:
                avi_out.write(frame)
            out_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            win_params['dr_min'] = cv2.getTrackbarPos(dr_min_text, window_name)
            win_params['dr_max'] = cv2.getTrackbarPos(dr_max_text, window_name)
            frame = frame.astype('float')
            frame[frame < win_params['dr_min']] = win_params['dr_min']
            frame[frame > win_params['dr_max']] = win_params['dr_max']
            frame -= win_params['dr_min']
            frame *= 255 / (win_params['dr_max'] - win_params['dr_min'])
            frame = frame.astype('uint8')

            if mode == 'analyze':
                frame = np.hstack([frame, self.plot])

            cv2.putText(frame, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color)
            cv2.imshow(window_name, frame)
            return out_frame, win_params

        prepare_window(win_params, mode='prepare')
        start_text = 'start analysis!'
        cv2.createTrackbar(start_text, window_name, 0, 1, lambda x: True if x == 0 else False) # finish prepare phase when set to 1

        logging.info('now preparing')
        prev_time = time.time()
        while True:
            time_d = 1 / self.fps
            while time.time() - prev_time < time_d:
                pass
            prev_time = time.time()
            frame, win_params = show_next_frame('prepareing...', win_params, mode='prepare')
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getTrackbarPos(start_text, window_name):
                break

        h, w = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        avi_out = cv2.VideoWriter(out_file_name + '.avi', fourcc, 10.0, (w, h))

        logging.info('now recording for init')
        prepare_window(win_params, mode='initialize')
        init_Y = np.empty((init_batch,) + frame.shape)

        with h5py.File(out_file_name + '.mat', 'w') as f:
            f['initialize_first_frame_t'] = time.time()

        if self.params.get('online', 'init_method') == 'seeded':
            if seed_file is None:
                raise ValueError('Please input analyzed mat file.')
            with h5py.File(seed_file, 'r') as f:
                ds_factor = self.params.get('online', 'ds_factor')
                A_seed = f['A'][()].reshape(frame.shape[0], frame.shape[1], -1)
                A_seed = cv2.resize(A_seed, (frame.shape[1]//ds_factor, frame.shape[0]//ds_factor))
                A_seed = A_seed.reshape((-1, A_seed.shape[-1]), order='F')
                self.estimates.A = A_seed

        prev_time = time.time()
        for i in range(init_batch):
            time_d = 1 / self.fps
            while time.time() - prev_time < time_d:
                pass
            prev_time = time.time()
            frame, _ = show_next_frame('initialize', win_params, mode='initialize', text_color=(0, 0, 255), avi_out=avi_out)
            init_Y[i] = frame.copy()
            cv2.waitKey(1)

        with h5py.File(out_file_name + '.mat', 'a') as f:
            f['initialize_last_frame_t'] = time.time()

        logging.info('now initializing...')
        Y_init = caiman.base.movies.movie(init_Y.astype(np.float32))
        self.initialize_online(model_LN=model_LN, Y=Y_init)

        t = init_batch
        logging.info('now running CNMF-E')
        prepare_window(win_params, mode='analyze')
        t_online = []

        def get_resized_a(idx, normalize=True):
            a_i = self.estimates.A[:, idx]
            a_i = a_i.reshape((self.estimates.dims[1], self.estimates.dims[0])).T
            a_i = cv2.resize(a_i.toarray(), (w, h))
            if normalize:
                a_i = a_i / a_i.max() * 255
            return a_i

        def set_plot(t, with_demixed=True):

            def round_to_even(x):
                if x % 2 == 0:
                    return x
                else:
                    return x - 1
            half_w, half_h = w//2, h//2
            even_w, even_h = round_to_even(w), round_to_even(h)

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

        with h5py.File(out_file_name + '.mat', 'a') as f:
            f['cnmfe_first_frame_t'] = time.time()
            A_size = frame.shape[0] * frame.shape[1]
            f.create_dataset('A', (A_size, self.N), maxshape=(A_size, None))
            f.create_dataset('C', (self.N, 100), maxshape=(None, None))
            f.create_dataset('S', (self.N, 100), maxshape=(None, None))

        prev_time = time.time()
        self.comp_upd = []
        self.t_shapes:List = []
        self.t_detect:List = []
        self.t_motion:List = []
        self.t_stat:List = []
        while True:
            time_d = 1 / self.fps
            while time.time() - prev_time < time_d:
                pass
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()

            try:
                set_plot(t)
                if t % 100 == 0:
                    self.set_results(1, t)
                    self.estimates.noisyC = np.hstack(
                        (self.estimates.noisyC, np.zeros((self.estimates.noisyC.shape[0], 100))))
                    self.estimates.C_on = np.hstack(
                        (self.estimates.C_on, np.zeros((self.estimates.C_on.shape[0], 100))))
                elif t % 100 == 10:
                    if self.params.get('online', 'ds_factor') > 1:
                        neuron_num = self.estimates.A.shape[-1]
                        A = np.hstack([cv2.resize(self.estimates.A[:, i].reshape(self.estimates.dims, order='F').toarray(),
                                                frame.shape[::-1]).reshape(-1, order='F')[:,None] for i in range(neuron_num)])
                elif t % 100 == 20:
                    with h5py.File(out_file_name + '.mat', 'a') as f:
                        f['A'].resize(A.shape)
                        f['A'][()] = A
                elif t % 100 == 30:
                    with h5py.File(out_file_name + '.mat', 'a') as f:
                        f['C'].resize(self.estimates.C.shape)
                        f['C'][()] = self.estimates.C
                elif t % 100 == 40:
                    with h5py.File(out_file_name + '.mat', 'a') as f:
                        f['S'].resize(self.estimates.S.shape)
                        f['S'][()] = self.estimates.S

                frame, _ = show_next_frame(f'FPS: {fps:.4f}', win_params, mode='analyze', text_color=(0, 0, 255), avi_out=avi_out)
                t_online.append(self.fit_next_from_raw(frame, t, model_LN=model_LN))
                t += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                print(sys.exc_info())
        
        with h5py.File(out_file_name + '.mat', 'a') as f:
            f['cnmfe_last_frame_t'] = time.time()
        avi_out.release()
        return self

    def get_model_LN(self):
        if self.params.get('online', 'ring_CNN'):
            pass
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
            Y = caiman.base.movies.load(fls[0], subindices=slice(init_batch),
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

    def set_results(self, epochs, t):
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

if __name__ == "__main__":
    pass