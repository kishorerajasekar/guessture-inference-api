import torch
import torch.nn as nn
from model.defaults import device

class LSTM(nn.Module):
    def __init__(self, seq_length, input_size, hidden_size, n_layers, n_classes):
        """
        :param input_size: num of features of input
        """
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.n_layers, batch_first = True)
        self.fc = nn.Linear(self.hidden_size*seq_length, n_classes)
        self.drop = nn.Dropout(p=0.2)
    
    def forward(self, x):
        # 3-D: (n_layers, bath_size, hidden_size)
        h0 = torch.zeros((self.n_layers, x.shape[0], self.hidden_size)).to(device)
        c0 = torch.zeros((self.n_layers, x.shape[0], self.hidden_size)).to(device)
        # pass
        out, _ = self.lstm(x, (h0, c0))
        # classification layer on output of last time step
        out = out.reshape(out.shape[0], -1) # flatten
        out = self.fc(out)
        return out

    
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.LSTM = LSTM(seq_len, input_size, hidden_size, n_layers, n_classes).to(device)
        
    def forward_once(self, x):
        return self.LSTM(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2