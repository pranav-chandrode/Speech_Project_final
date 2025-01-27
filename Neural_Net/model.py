import torch
import torch.nn as nn
import torch.nn.functional as F

class DropNormAct(nn.Module):
    """ includes dropout, normalization and activation, used as a post-processing step after a 1D convolution layer"""
    def __init__(self, n_feats, dropout, keep_shape = False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape

    def forward(self, x):
        x = x.transpose(1,2)   # x -> [batch, feature, time] and we have normalize features not time, therefore shape after transpose x -> [batch, time, feature]
        x = self.dropout(F.gelu(self.norm(x)))

        if self.keep_shape:
            return x.transpose(1,2)
        else:
            return x
        

class SpeechRecognition(nn.Module):
    hyper_parameters = {
        "num_classes" : 29,
        "n_feats" : 81,
        "dropout" : 0.1,
        "hidden_size" : 1024,
        "num_layers" : 1
    }

    def __init__(self, num_classes, n_feats, dropout, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats,n_feats,10,2, padding=10//2),
            DropNormAct(n_feats=n_feats,dropout=dropout)
        )

        self.dense = nn.Sequential(
            nn.Linear(n_feats,128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128,128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers,dropout=0.0)
        
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size,num_classes)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),  # n * 1 -> 1 for non-bidirectional lstm
                torch.zeros(n*1, batch_size, hs))
    

    def forward(self, x, hidden):
        # print("initial shape : ",x.shape)
        x = x.squeeze(1)  # spectrogram shape [batch,1,81,time] so need to squeeze it
        # print("after squeeze : ",x.shape)
        x = self.cnn(x)   # (input) [batch, feature, time] -> (output) [batch, time, feature]
        # print("after cnn :",x.shape)
        x = self.dense(x) # [batch, time, feature]
        # print("after dense : ",x.shape)
        x = x.transpose(0,1) # [time, batch, feature]
        # print("after transpose (0,1)",x.shape)
        out, (hn,cn) = self.lstm(x, hidden)
        # print("\"out\" shape : ",out.shape)
        x = self.dropout_layer(F.gelu(self.layer_norm2(out)))  # [time, batch, feature]
        # print("after dropout : ",x.shape)
        return self.final_fc(x), (hn,cn)
    
        
