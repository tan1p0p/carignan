import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import joblib
from cnmfereview.utils import crop_footprint, process_traces

from .res_block import CNN
from .extractor import Extractor

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_cnn_for_trace=True):
        super().__init__()
        self.use_cnn_for_trace = use_cnn_for_trace
        self.conv = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if self.use_cnn_for_trace:
            x = self.conv(x)
        _, (hn, _) = self.lstm(x)
        return F.relu(self.fc(hn.squeeze(0)))

class DeepModel(nn.Module):
    def __init__(self, s_stage):
        super().__init__()

        if s_stage == 'CNN':
            self.spatial_stage = CNN(block_num=3)
        elif s_stage == 'ShufflenetV2':
            self.spatial_stage = Extractor('shufflenet_v2_x1_0')

        self.temporal_stage = CNN_LSTM(500, hidden_dim=500, output_dim=500)

        s_out_len = torch.numel(self.spatial_stage(torch.rand((1, 1, 80, 80))))
        self.out = nn.Sequential(
            nn.Linear(s_out_len + 500, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def norm(self, x, dim=None):
        min_v, _ = x.min(dim=dim, keepdims=True)
        max_v, _ = x.max(dim=dim, keepdims=True)
    
        # to avoid zero division
        zero_idx = max_v == min_v
        max_v[zero_idx] = 1
        min_v[zero_idx] = 0
    
        result = (x - min_v) / (max_v - min_v)
        return result

    def forward(self, inputs):
        t, s = inputs
        t = self.norm(t, 2)
        t = self.temporal_stage(t)
        s_shape = s.shape
        s = self.norm(s.reshape(-1, 1, 80 * 80), 2).reshape(s_shape)
        s = torch.flatten(self.spatial_stage(s), 1)
        return self.out(torch.cat((s, t), 1))

class FpDetector():
    def __init__(self, method, model_dir='./models/'):
        self.method = method
        if method == None:
            pass
        elif method == 'TPOT':
            self.model = joblib.load(os.path.join(model_dir, 'cr_tutorial_tpot.joblib'))
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            if method == 'CNN':
                model_name = 'CNN.pth'
            elif method == 'ShufflenetV2':
                model_name = 'shufflenet_v2_x1_0.pth'
            else:
                raise ValueError(f'{method} detection is unsupported.')
            self.model = DeepModel(method)
            self.model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
            self.model.to(self.device).eval()

    def predict(self, unchecked_A, unchecked_C):
        spatial = crop_footprint(unchecked_A, 80)
        trace = process_traces(unchecked_C, 500)
        if self.method == 'TPOT':
            spatial = spatial.reshape((spatial.shape[0], -1))
            combined = np.concatenate((spatial, trace), axis=1)
            return self.model.predict(combined), 0.5
        else:
            trace = trace.reshape(-1, 1, 500)
            spatial = spatial.reshape(-1, 1, 80, 80)
            pred = self.model((torch.from_numpy(trace).float(), torch.from_numpy(spatial).float()))
            return pred[:, 0], 0.5
