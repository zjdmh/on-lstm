"""
All model structure

<<<<<<<< HEAD
Author: Qingliang Li 
12/23/2022 - V1.0  LSTM, CNN, ConvLSTM edited by Qingliang Li
.........  - V2.0
.........  - V3.0
"""
import math
import torch
import torch.nn as nn
import numpy as np
from convlstm import ConvLSTM
# ------------------------------------------------------------------------------------------------------------------------------
'''class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于为序列数据添加位置信息。
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)  # 确保使用 float32
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x'''
# simple lstm model with fully-connect layer
class LSTMModel(nn.Module):
    """single task model"""

    '''
    def __init__(self, cfg, lstmmodel_cfg):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(lstmmodel_cfg["input_size"], lstmmodel_cfg["hidden_size"], batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        # 添加一个隐藏层
        self.hidden_layer = nn.Linear(lstmmodel_cfg["hidden_size"], 64)  # 隐藏层大小为64
        self.hidden_activation = nn.ReLU()  # 使用ReLU激活函数
        self.dense = nn.Linear(64, lstmmodel_cfg["out_size"])  # 输出层的输入大小改为64

    def forward(self, inputs, aux):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        # 只取LSTM输出的最后一个时间步
        x = x[:, -1, :]
        # 通过隐藏层
        x = self.hidden_layer(x)
        x = self.hidden_activation(x)  # 应用激活函数
        # 通过输出层
        x = self.dense(x)
        return x'''

    def __init__(self, cfg,lstmmodel_cfg):
        super(LSTMModel,self).__init__()
        self.lstm = nn.LSTM(lstmmodel_cfg["input_size"], lstmmodel_cfg["hidden_size"],batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(lstmmodel_cfg["hidden_size"],lstmmodel_cfg["out_size"])

    def forward(self, inputs,aux):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        # we only predict the last step
        x = self.dense(x[:,-1,:]) 
        return x
# ------------------------------------------------------------------------------------------------------------------------------
# simple CNN model with fully-connect layer
class CNN(nn.Module):
    """single task model"""

    def __init__(self, cfg):
        super(CNN,self).__init__()
        self.latn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1
        self.lonn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1
        self.cnn = nn.Conv2d(in_channels=cfg["input_size_cnn"],out_channels=cfg["hidden_size"],kernel_size=cfg["kernel_size"],stride=cfg["stride_cnn"])
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(int(cfg["hidden_size"])*int(self.latn)*int(self.lonn),1)

    def forward(self, inputs,aux):
        x = self.cnn(inputs.float())
        x = self.drop(x)
        x = x.reshape(x.shape[0],-1)
        # we only predict the last step
        x = self.dense(x) 
        return x
# ------------------------------------------------------------------------------------------------------------------------------
# simple convlstm model with fully-connect layer
class ConvLSTMModel(nn.Module):
    """single task model"""

    def __init__(self, cfg):
        super(ConvLSTMModel,self).__init__()
        self.ConvLSTM_net = ConvLSTM(input_size=(int(2*cfg["spatial_offset"]+1),int(2*cfg["spatial_offset"]+1)),
                       input_dim=int(cfg["input_size"]),
                       hidden_dim=[int(cfg["hidden_size"]), int(cfg["hidden_size"]/2)],
                       kernel_size=(int(cfg["kernel_size"]), int(cfg["kernel_size"])),
                       num_layers=2,cfg=cfg,batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(int(cfg["hidden_size"]/2)*int(2*cfg["spatial_offset"]+1)*int(2*cfg["spatial_offset"]+1),1)
        #self.batchnorm = nn.BatchNorm1d(int(cfg["hidden_size"]/2))

    def forward(self, inputs,aux,cfg):
        threshold = torch.nn.Threshold(0., 0.0)
        inputs_new = torch.cat([inputs, aux], 2).float()
        #inputs_new = inputs.float()
        hidden =  self.ConvLSTM_net.get_init_states(inputs_new.shape[0])
        last_state, encoder_state =  self.ConvLSTM_net(inputs_new.clone(), hidden)
        last_state = self.drop(last_state)
        #Convout = last_state[:,-1,:,cfg["spatial_offset"],cfg["spatial_offset"]]
        Convout = last_state[:,-1,:,:,:]
        #Convout = self.batchnorm(Convout)
        shape=Convout.shape[0]
        #print('Convout shape is',Convout.shape)
        Convout=Convout.reshape(shape,-1)
        Convout = torch.flatten(Convout,1)
        Convout = threshold(Convout)
        predictions=self.dense(Convout)

        return predictions



