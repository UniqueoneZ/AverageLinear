'''
A complete implementation version containing all code (including ablation components)
'''
import numpy as np
import torch
import torch.nn as nn
from layers.RevIN import RevIN
from einops import rearrange, repeat, einsum
import torch.nn.functional as F

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

        self.rnn_type = configs.rnn_type
        self.dec_way = configs.dec_way
        self.seg_len = configs.seg_len
        self.channel_id = configs.channel_id
        self.revin = configs.revin
        self.d_state = configs.d_state
        #we can get the feature_num directly
        #however, this features stands for the task label, it's not the same idea as we need the number of the variate


        self.seg_num_x = self.seq_len//self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        if self.revin:
            self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=False)

        self.predict = nn.Sequential(
            nn.Linear(self.seg_num_x, self.seg_num_y),
        )



    def forward(self, x):

        #x shape: b,l,c
        #normalize
        if self.revin:
            x = self.revinLayer(x, 'norm')
        else:
            means = x.mean(1, keepdim = True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim = 1, keepdim = True, unbiased = False) + 1e-5)
            x /= stdev
        x = x.permute(0,2,1)# b,c,l
        x = x.reshape(x.shape[0], self.enc_in, self.seg_num_x, self.seg_len) # b, c, n, w
        x = torch.mean(x, dim = 3)#b,c,n

        y = self.predict(x) #b,c,m
        #denorm
        y=y.permute(0,2,1) #b,m,c
        if self.revin:
            y = self.revinLayer(y, 'denorm')
        else:
            #y.shape : b,m,c
            y = y * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seg_num_y, 1))
            y = y + (means[:, 0, :].unsqueeze(1).repeat(1, self.seg_num_y, 1))
        return y

