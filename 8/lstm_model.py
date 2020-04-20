import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 10)
        
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        _, (h_out1, _) = self.lstm1(x, (h_0, c_0))
        h_out1 = h_out1.view(-1, self.hidden_size)
        out1 = self.fc1(h_out1)
        out1 = out1.reshape((out1.shape[0], out1.shape[1], 1))
        
        h_1 = Variable(torch.zeros(
            self.num_layers, out1.size(0), self.hidden_size))
        
        c_1 = Variable(torch.zeros(
            self.num_layers, out1.size(0), self.hidden_size))
        
        _, (h_out2, _) = self.lstm2(out1, (h_1, c_1))
        h_out2 = h_out2.view(-1, self.hidden_size)
        
        out2 = self.fc2(h_out2)
        
        return out2