import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length=5000, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * torch.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0) # [5000, 1, d_model],so need seq-len <= 5000
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        x += self.pe[:, :x.size(1), :].repeat(x.shape[0],1, 1)
        
        return self.dropout(x)
    
    
class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1), dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class Transpose(nn.Module):
    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, x):
        
        return x.transpose(self.dim1, self.dim2)


class AdaptiveConcatPoolRNN(nn.Module):
    def __init__(self, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
    def forward(self,x):
        #input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)

        if(self.bidirectional is False):
            t3 = x[:,:,-1]
        else:
            channels = x.size()[1]
            t3 = torch.cat([x[:,:channels,-1],x[:,channels:,0]],1)
        out=torch.cat([t1.squeeze(-1),t2.squeeze(-1),t3],1) #output shape bs, 3*ch
        return out


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



class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=2, num_layers=2):
        super(TransformerEncoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = self.encoder(x)
        
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, d_model=64, nhead=2, num_layers=2):
        super(TransformerDecoder, self).__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x, memory, mask=None):
        if torch.is_tensor(mask):
            x = self.decoder(x, memory, tgt_mask=mask)
        else:
            x = self.decoder(x, memory)
        
        return x
    

class Transformer(nn.Module):
    def __init__(
        self,
        batch_size,
        input_channels,
        seq_length,
        num_classes,
        d_model,
        nhead,
        num_layers,
        dropout=0.0,
        decode=True,
        masking=False,
        clf_pool=True,
    ):
        super(Transformer, self).__init__()
        
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
        
        self.clf_pool = clf_pools(clf_pool, input_dim=self.d_model, seq_length=seq_length)

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
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)
