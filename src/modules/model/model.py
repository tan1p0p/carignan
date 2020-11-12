import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from res_block import ResBlock
from san_block import SAN

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv1D(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)
        output, (hn, cn) = self.lstm(x)
        hn = hn.squeeze(0)
        return F.relu(self.fc(hn))

class ResNet_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.SpatialStage = ResNet()
        self.TemporalStage = CNN_LSTM()
        self.out = nn.Linear()

    def forward(spatial, temporal):
        s = self.SpatialStage(spatial)
        t = self.TemporalStage(temporal)
        return F.sigmoid(self.out(torch.cat(s, t)))

class SANet_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.SpatialStage = SAN()
        self.TemporalStage = CNN_LSTM()
        self.out = nn.Linear()

    def forward(spatial, temporal):
        s = self.SpatialStage(spatial)
        t = self.TemporalStage(temporal)
        return F.sigmoid(self.out(torch.cat(s, t)))

def get_model(with_spatial:bool=True, use_cnn_for_trace:bool=True,
              frame:int=500, crop_size:int=80):
    # in_out_neurons = 1 # use raw data

    t_input = layers.Input(shape=(frame, 1))
    if use_cnn_for_trace:
        t = layers.Conv1D(filters=64, kernel_size=7, padding='same', activation='relu')(t_input)
        t = layers.MaxPooling1D(pool_size=2)(t)
    else:
        t = t_input

    t = layers.LSTM(
        100,
        # batch_input_shape=(None, int(frame/2), in_out_neurons),
        return_sequences=False)(t)

    if with_spatial:
        s_input = layers.Input(shape=(crop_size, crop_size, 1))
        s = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(s_input)
        s = layers.MaxPooling2D(2, strides=2, padding="same")(s)
        s = layers.Dropout(0.5)(s)
        s = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(s)
        s = layers.MaxPooling2D(2, strides=2, padding="same")(s)
        s = layers.Flatten()(s)
        x = layers.Concatenate(axis=1)([t, s])
    else:
        x = t
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    if with_spatial:
        model = models.Model(inputs=[t_input, s_input], output=out)
    else:
        model = models.Model(inputs=t_input, output=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
