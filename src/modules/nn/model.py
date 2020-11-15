import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .res_block import ResNet
from .san_block import SAN

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

class Model(nn.Module):
    def __init__(self,
                 s_stage, s_size=(80, 80), res_block_num=None, san_layers=None, san_kernels=None,
                 t_size=500, t_hidden_dim=300, t_output_dim=300, use_cnn_for_trace=True):
        super().__init__()

        if s_stage == 'ResNet':
            if res_block_num is None:
                raise ValueError('ResNet needs res_block_num')
            self.spatial_stage = ResNet(block_num=res_block_num)

        elif s_stage == 'SAN':
            if san_layers is None or san_kernels is None:
                raise ValueError('SAN needs san_layers and san_kernels')
            self.spatial_stage = SAN(sa_type=0, layers=san_layers, kernels=san_kernels)

        self.temporal_stage = CNN_LSTM(
            t_size,
            hidden_dim=t_hidden_dim,
            output_dim=t_output_dim,
            use_cnn_for_trace=use_cnn_for_trace)

        self.spatial_stage.cuda()
        s_in = torch.rand((1, 1) + s_size).cuda()
        s_out_size = self.spatial_stage(s_in).shape
        s_out_len = s_out_size[1] * s_out_size[2] * s_out_size[3]

        t_out_size = self.temporal_stage(torch.rand(1, 1, t_size)).shape
        t_out_len = t_out_size[1]

        print(f'spatial feature len: {s_out_len}, temporal feature len: {t_out_len}')
        self.out = nn.Sequential(
            nn.Linear(s_out_len + t_out_len, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        t, s = inputs
        t = self.temporal_stage(t)
        s = torch.flatten(self.spatial_stage(s), 1)
        return self.out(torch.cat((s, t), 1))
