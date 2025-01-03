import numpy as np
import torch
import torch.nn as nn
from layers.RevIN import RevIN
from einops import rearrange, repeat, einsum
from models.model import *
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(features, 2 * features, bias=False)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(2 * features, features, bias=False)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.activation(x1)
        x1 = self.linear2(x1)
        return x + x1

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.batch_size = configs.batch_size
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.channel_id = configs.channel_id
        self.revin = configs.revin
        self.is_training = configs.is_training
        
        
        self.elements_to_remove =  [[1, 2, 3, 5, 6, 8, 9, 10, 19], [4, 7], [11, 12], [16, 17, 18]]
        self.elements_to_remove1 = [0, 13, 14, 15, 20]
        

        #对于所有的周期模式以及周期函数都设置独立的预测函数
        if self.channel_id:
            self.predict_layers =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.seq_len, self.d_model, bias = False),  # 输出层
                ResidualBlock(self.d_model),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.pred_len, bias = False)
            ) for _ in range(len(self.elements_to_remove))
            ])
            self.predict_layers1 =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.seq_len, self.d_model, bias = False),  # 输出层
                ResidualBlock(self.d_model),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.pred_len, bias = False)
            ) for _ in range(len(self.elements_to_remove1))
            ])

        #定义拓展的通道数
        self.channel_number = configs.c_layers

        #定义一个通道层
        self.predict_layers_list = nn.ModuleList([
            nn.ModuleList([  # 内部再使用 nn.ModuleList
                nn.Sequential(
                    nn.Linear(self.seq_len, self.d_model, bias=False),  # 输出层
                    ResidualBlock(self.d_model),
                    nn.SiLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model, self.pred_len, bias=False)
                ) for _ in range(len(self.elements_to_remove))
            ]) for i in range(self.channel_number)
        ])
        self.predict_layers_list1 = nn.ModuleList([
            nn.ModuleList([  # 内部再使用 nn.ModuleList
                nn.Sequential(
                    nn.Linear(self.seq_len, self.d_model, bias=False),  # 输出层
                    ResidualBlock(self.d_model),
                    nn.SiLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model, self.pred_len, bias=False)
                ) for _ in range(len(self.elements_to_remove1))
            ]) for i in range(self.channel_number)
        ])


        self.channel_layer = nn.ModuleList([
            nn.Sequential(
            nn.Linear(self.enc_in, 2 * self.enc_in, bias = False),  # 输出层
            ResidualBlock(2 * self.enc_in),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.enc_in, self.enc_in, bias = False)
        ) for _ in range(self.channel_number)
        ])

        
        if self.revin:
            self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=False)



    def forward(self, x):
        (batch_size,_,_) = x.shape
        # normalize
        if self.revin:
            x = self.revinLayer(x, 'norm').permute(0, 2, 1)
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last).permute(0, 2, 1) # b,c,s
        
        x_clone = []
        for i in range(self.channel_number):
            x1 = x.clone()
            x_clone.append(self.channel_layer[i](x1.permute(0,2,1)).permute(0,2,1))
        
        #定义原始数据预测出来的结果
        y = torch.zeros(batch_size, self.enc_in, self.pred_len).to(x.device)
        #定义通独通关
        for i in range(len(self.elements_to_remove)):
            y[:,self.elements_to_remove[i],:] = self.predict_layers[i](x[:,self.elements_to_remove[i],:])
        for i in range(len(self.elements_to_remove1)):
            y[:,self.elements_to_remove1[i],:] = self.predict_layers1[i](x[:,self.elements_to_remove1[i],:])

        #定义拓展数据预测出来的结果
        y_list = []
        for i in range(self.channel_number):
            y_list.append(torch.zeros(batch_size, self.enc_in, self.pred_len).to(x.device))
        
        #预测结果
        for i in range(self.channel_number):
            for j in range(len(self.elements_to_remove)):
                y_list[i][:,self.elements_to_remove[j],:] = self.predict_layers_list[i][j](x_clone[i][:,self.elements_to_remove[j],:])
        for i in range(self.channel_number):
            for j in range(len(self.elements_to_remove1)):
                y_list[i][:,self.elements_to_remove1[j],:] = self.predict_layers_list1[i][j](x_clone[i][:,self.elements_to_remove1[j],:])
    
        #累加结果
        for i in range(self.channel_number):
            y += y_list[i]/self.channel_number
        
        y = y/2

        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last


        return y

