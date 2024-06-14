import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch import Tensor
from torch.autograd import Variable
from dl_models.transformer import *

#egg_path = os.path.join(os.getcwd(),  "code/extensions/cauchy/cauchy_mult-0.0.0-py3.9-linux-x86_64.egg")
#sys.path.append(egg_path)
#egg_path = os.path.join(os.getcwd(),  "extensions/cauchy/cauchy_mult-0.0.0-py3.9-linux-x86_64.egg")
#egg_path = "~/ssm_ecg/code/extensions/cauchy/cauchy_mult-0.0.0-py3.9-linux-x86_64.egg"
#egg_path = "/Users/simonjaxy/Documents/vub/WP1/ssm_ecg_github/code/extensions/cauchy/cauchy_mult-0.0.0-py3.9-linux-x86_64.egg"
#print(egg_path)
#sys.path.append(egg_path)

#import cauchy_mult
from dl_models.s4 import S4
from .basic_conv1d import bn_drop_lin
from dl_models.rff import RFFModel, ConvRFFModel

###########################################################################################
###########################################################################################
###########################################################################################


def calc_out_conv(len_, kz, p, s):
    return math.floor(((len_ + 2*p - 1*(kz-1) -1) / s) + 1)
def calc_out_pool(len_, kz, p, s, pool):
    if pool == 'max':
        return math.floor(((len_ + 2*p - 1*(kz-1) -1) / s) + 1)
    else:
        return math.floor(((len_ + 2*p - 1*(kz)) / s) + 1)


def act_funs(act_fun):
    if act_fun == 'relu':
        return nn.ReLU()
    elif act_fun == 'gelu':
        return nn.GELU()
    elif act_fun == 'elu':
        return nn.ELU()
    else:
        raise NotImplemented("Currently no other activation functions under consideration")


def poolings(pool):
    if pool == 'max':
        return nn.MaxPool1d
    elif pool == 'avg':
        return nn.AvgPool1d
    elif pool == 'None':
        return None
    else:
        raise NotImplemented("Currently no other activation functions under consideration")


def normalizations(norm):
    if norm == 'BN':
        return nn.BatchNorm1d
    elif norm == 'LN':
        return nn.LayerNorm
    elif norm == 'None':
        return None
    else:
        raise NotImplemented("Currently no other activation functions under consideration")
    
    
def dropouts(dropout):
    if dropout == "None":
       return nn.Identity()
    else:
       return nn.Dropout(p=float(dropout))



def weight_init(m):
    '''call weight initialization for model n via n.appy(weight_init)'''
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
        

###########################################################################################
###########################################################################################
###########################################################################################


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=-1)

        return x


class LSTM_tuple_to_single(nn.Module):
    def __init__(self, output=False):
        super().__init__()
        self.output = output

    def forward(self, t):
        out, (h, c) = t

        if self.output:
            return out
        else:
            return h[-1]


class LSTM(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(LSTM, self).__init__()

        input_size = int(config["input_size"])
        num_classes = int(config["num_classes"])
        hidden_size = int(config["hidden_size"])
        self.hidden_size = hidden_size
        num_layers = int(config["num_layers"])
        self.num_layers = num_layers
        bias = bool(config["lstm_bias"])
        dropout = float(config["lstm_dropout"])
        bidirectional = bool(config["bidirectional"])
        self.bidirectional = bidirectional
        act_fun = act_funs(str(config["act_fun"]))
        norm = str(config["normalization"])
        normalization = normalizations(norm)

        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bias=bias, dropout=dropout, bidirectional=bidirectional)

        layers = [LSTM_tuple_to_single(), nn.Flatten()]
        if normalization:
            ### currently not implemented
            if norm == "LN":
                layers.append(
                    normalization([hidden_size])
                )
            else:
                layers.append(
                    normalization(hidden_size)
                )
        layers.append(act_fun)
        layers.append(nn.Linear(hidden_size, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #out = self.layers(x)
        d = 2 if self.bidirectional else 1
        h_0 = Variable(torch.zeros(d*self.num_layers, x.shape[0], self.hidden_size).to(x.device))
        c_0 = Variable(torch.zeros(d*self.num_layers, x.shape[0], self.hidden_size).to(x.device))
        out = self.LSTM(x.transpose(1, 2), (h_0, c_0))
        out = self.layers(out)

        return out



###########################################################################################
###########################################################################################
###########################################################################################


class CNN(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(CNN, self).__init__()

        in_features = config["in_features"]
        num_classes = config["num_classes"]
        in_length = config["in_length"]
        num_layers = config["num_layers"]
        feature_sizes = config["feature_sizes"]
        kernel_sizes = config["kernel_sizes"]
        padding = config["padding"]
        strides = config["strides"]
        act_fun = act_funs(config["act_fun"])
        pool = config["pool"]
        pool_fn = poolings(pool)
        norm = config["normalization"]
        normalization = normalizations(norm)
        dropout = dropouts(config["dropout"])
        self.global_pool = config["global_pool"]

        if isinstance(feature_sizes, int) or isinstance(feature_sizes, np.int64):
            feature_sizes = [feature_sizes]*num_layers
        if isinstance(kernel_sizes, int) or isinstance(kernel_sizes, np.int64):
            kernel_sizes = [kernel_sizes]*num_layers
        if isinstance(strides, int) or isinstance(strides, np.int64):
            strides = [strides]*num_layers

        layers = [
            nn.Conv1d(in_features, feature_sizes[0], kernel_sizes[0], padding=padding, stride=strides[0]),
            act_fun,
            dropout,
        ]
        out_length = calc_out_conv(in_length, kernel_sizes[0], padding, strides[0])

        
        if pool_fn:
            layers.append(
                pool_fn(kernel_size=kernel_sizes[0], padding=padding, stride=strides[0])
            )
            out_length = calc_out_pool(out_length, kernel_sizes[0], padding, strides[0], pool)
                                                                                       

        if normalization:
            if norm == "LN":
                layers.append(
                    normalization([feature_sizes[0], out_length])
                )
            else:
                layers.append(
                    normalization(feature_sizes[0])
                )

        for i in range(1, num_layers):
            layers.append(
                nn.Conv1d(feature_sizes[i-1], feature_sizes[i], kernel_sizes[i], padding=padding, stride=strides[i])
            )
            layers.append(
                act_fun,
            )
            out_length = calc_out_conv(out_length, kernel_sizes[i], padding, strides[i])
            if pool_fn:
                layers.append(
                pool_fn(kernel_size=kernel_sizes[i], padding=padding, stride=strides[i])
                )
                out_length = calc_out_pool(out_length, kernel_sizes[i], padding, strides[i], pool)
            if normalization:
                if norm == "LN":
                    layers.append(
                        normalization([feature_sizes[i], out_length])
                    )
                else:
                    layers.append(
                        normalization(feature_sizes[i])
                    )
            layers.append(dropout)

        #layers.append()
        self.layers = nn.Sequential(*layers)
        if self.global_pool:
            self.fc = nn.Linear(out_length, num_classes)
        else:
            self.fc = nn.Linear(feature_sizes[-1]*out_length, num_classes)
            
        
    def forward(self, x):
        bz = x.shape[0]
        x = self.layers(x)
        if self.global_pool:
            x = x.mean(dim=1)
        else:
            x = x.reshape(bz, -1)
        out = self.fc(x)

        return out


###########################################################################################
###########################################################################################
###########################################################################################



class S4Model(nn.Module):

    def __init__(
        self,
        config,
    ):
        d_input=config['d_input']
        d_output=config['num_classes']
        d_model=config['d_model']
        d_state=config['d_state']
        n_layers=config['num_layers']
        dropout=config['dropout']
        prenorm=False
        l_max=config['l_max']
        transposed_input=True
        normalization = config['normalization']
        bidirectional=config['bidirectional']
        use_meta_information_in_head=False
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.transposed_input = transposed_input
        self.encoder = nn.Conv1d(
            d_input, d_model, 1) if transposed_input else nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(
                    d_model=d_model,
                    l_max=l_max,
                    bidirectional=bidirectional,
                    postact='glu',
                    dropout=dropout,
                    transposed=True,
                    d_state=d_state
                )
            )
            if normalization == "BN":
                self.norms.append(nn.BatchNorm1d(d_model))
            elif normalization == "LN":
                self.norms.append(nn.LayerNorm(d_model))
            else:
                self.norms.append(nn.Identity())
            self.normalization = normalization
            self.dropouts.append(nn.Dropout2d(dropout))

        # Linear decoder
        # MODIFIED TO ALLOW FOR MODELS WITHOUT DECODER
        if(d_output is None):
            self.decoder = None
        else:
            self.decoder = nn.Linear(d_model + 64 if use_meta_information_in_head else d_model, d_output)
        if use_meta_information_in_head:
            meta_modules = bn_drop_lin(7, 64, bn=False,actn=nn.ReLU()) +\
            bn_drop_lin(64, 64, bn=True, p=0.5, actn=nn.ReLU()) + bn_drop_lin(64, 64, bn=True, p=0.5, actn=nn.ReLU())
            self.meta_head = nn.Sequential(*meta_modules)


    def forward(self, x, rate=1.0):
        
        #Input x is shape (B, d_input, L) if transposed_input else (B, L, d_input)
        
        x = self.encoder(
            x)  # (B, d_input, L) -> (B, d_model, L) if transposed_input else (B, L, d_input) -> (B, L, d_model)

        if(self.transposed_input is False):
            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                if self.normalization == "LN":
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)
                else:
                    z = norm(z)

            # Apply S4 block: we ignore the state input and output

            z, _ = layer(z, rate=rate)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                if self.normalization == "LN":
                    x = norm(x.transpose(-1, -2)).transpose(-1, -2)
                else:
                    x = norm(x)
        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        if self.decoder is not None:
            x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

    def forward_with_meta(self, x, meta_feats, rate=1.0):
        
        #Input x is shape (B, d_input, L) if transposed_input else (B, L, d_input)
    
        x = self.encoder(
            x)  # (B, d_input, L) -> (B, d_model, L) if transposed_input else (B, L, d_input) -> (B, L, d_model)

        if(self.transposed_input is False):
            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output

            z, _ = layer(z, rate=rate)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        meta_feats = self.meta_head(meta_feats)
        x = torch.cat([x, meta_feats], axis=1)

        # Decode the outputs
        if self.decoder is not None:
            x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

###########################################################################################
###########################################################################################
###########################################################################################



class Transpose(nn.Module):
    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):

        return x.transpose(self.dim1, self.dim2)


def clf_pools(clf_pool, input_dim=512, seq_length=250):
    if clf_pool == 'self_attention':
        return SelfAttentionPooling(input_dim=input_dim)
    elif clf_pool == 'adaptive_concat_pool':
        return nn.Sequential(
            Transpose(dim1=1, dim2=2),
            AdaptiveConcatPoolRNN(),
        )
    elif clf_pool == 'mean':
        return nn.AvgPool1d(kernel_size=seq_length)
    else:
        return nn.Flatten()


class Transformer(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(Transformer, self).__init__()

        batch_size = config["batch_size"]
        input_channels = config["input_channels"]
        seq_length = config["seq_length"]
        num_classes = config["num_classes"]
        d_model = config["d_model"]
        nhead = config["nhead"]
        num_layers = config["num_layers"]
        dropout = config["dropout"]
        decode = config["decode"]
        masking = config["masking"]
        clf_pool = config["clf_pool"]


        self.decode = decode
        self.masking = masking
        self.seq_length = seq_length
        self.d_model = d_model

        self.input_fc = nn.Linear(input_channels, self.d_model)

        self.pos_encoding = PositionalEncoding(self.d_model, self.seq_length, dropout)

        # object queries as learnable parameters: 
        # https://www.sciencedirect.com/science/article/pii/S0010482522001172
        self.tgt = nn.Parameter(
            # necessary? since we learn it anyway
            data=torch.randn(batch_size, self.seq_length,self.d_model), #self.pos_encoding(),
            requires_grad=True
        )

        self.encoder = TransformerEncoder(self.d_model, nhead, num_layers)
        self.decoder = TransformerDecoder(self.d_model, nhead, num_layers)\
            if self.decode else nn.Identity()

        self.clf_pool = clf_pools(clf_pool, input_dim=self.d_model, seq_length=self.seq_length)

        if clf_pool == 'self_attention':
            self.output_fc = nn.Linear(self.d_model, num_classes)
        elif clf_pool == 'adaptive_concat_pool':
            self.output_fc = nn.Linear(3*self.d_model, num_classes)
        elif clf_pool == 'mean':
            self.output_fc = nn.Linear(self.d_model, num_classes)
        else:
            self.output_fc = nn.Linear(self.seq_length*self.d_model, num_classes)

    def forward(self, x):
        # input: (N, C, Seq_Len)
        x = x.transpose(1, 2)
        # embedding: (N, Seq_len, d_model)
        x = self.input_fc(x)
        # pos encoding: (N, Seq_len, d_model)
        x = self.pos_encoding(x)
        #self.tgt = self.pos_encoding(self.tgt)
        # encode: (N, Seq_len, d_model)
        x = self.encoder(x)

        # decode: (N, Seq_len, d_model)
        if self.decode:
            # use masking to hide inps
            if self.masking:
                tgt_mask = self.get_tgt_mask(self.seq_length).to(x.device)
                x = self.decoder(self.tgt, x, tgt_mask)
            else:
                x = self.decoder(self.tgt, x)
        else:
            x = self.decoder(x)

        # if attn pool: (N, d_model)
        # else: seq collapse: x = x.reshape(-1, self.seq_length*self.d_model) - > (N, Seq_len*d_model)
        # alternative: x.mean(dim=1).squeeze() -> (N, d_model)
        x = self.clf_pool(x)
        # classify: (N, num_classes)
        x = self.output_fc(x)

        return x.squeeze() 


    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a square matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        return mask


    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token) 


###########################################################################################
###########################################################################################
###########################################################################################

class RFF(RFFModel):
    def __init__(self, config):
        kernel=config['kernel']
        mc=config['mc']
        num_layers=config['num_layers']
        in_dims=config['in_dims']
        N_RFs=config['N_RFs']
        num_classes = config["num_classes"]

        super(RFF, self).__init__(
                kernel,
                mc,
                num_layers,
                in_dims,
                N_RFs,
                num_classes,
        )


class ConvRFF(ConvRFFModel):
    def __init__(self, config):
        kernel=config['kernel']
        mc=config['mc']
        num_layers=config['num_layers']
        in_channels=config['in_channels']
        feature_sizes=config['feature_sizes'] 
        num_classes = config["num_classes"]
        length=config['length']
        global_pool=config['global_pool']
        kernel_size=config['kernel_size'] 
        stride=config['stride']
        padding=config['padding']
        group=config['group']  

        super(ConvRFF, self).__init__(
                kernel,
                mc,
                num_layers,
                in_channels,
                feature_sizes, 
                num_classes,
                length,
                global_pool,
                kernel_size, 
                stride,
                padding,
                group
        )
