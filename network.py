import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.rnn = nn.RNN(input_size, 
                          hidden_size, 
                          1,
                          batch_first = True,
                         )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        hidden = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, hidden = self.rnn(x, hidden)
        out = self.relu(out[:,-1,:])
        out = self.fc(out)
        return out
        
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.rnn = nn.LSTM(input_size, 
                     hidden_size, 
                     1,
                     batch_first = True,
                     )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        hidden = (torch.zeros(1, x.size(0), self.hidden_size).to(x.device), torch.zeros(1, x.size(0), self.hidden_size).to(x.device))
        out, hidden = self.rnn(x, hidden)
        out = self.relu(out[:,-1,:])
        out = self.fc(out)
        return out
    
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.rnn = nn.LSTM(input_size, 
                     hidden_size, 
                     1,
                     bidirectional=True,
                     batch_first = True
                     )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2 * hidden_size, 20)
        self.fc2 = nn.Linear(20, output_size)
        
    def forward(self, x):
        hidden = (torch.zeros(2, x.size(0), self.hidden_size).to(x.device), torch.zeros(2, x.size(0), self.hidden_size).to(x.device))
        out, hidden = self.rnn(x, hidden)
        out = self.fc1(out[:,-1,:])
        out = self.relu(out)
        out = self.fc2(out)
        return torch.sigmoid(out)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.relu(out)
        out = self.fc(out)
        return out