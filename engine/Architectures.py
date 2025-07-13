import torch
import torch.nn as nn
import math

# Transformer architecture
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class SleepStageTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=2, num_layers=2, dropout=0.1):
        super(SleepStageTransformer, self).__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.input_proj(x)               # [batch, seq, hidden_dim]
        x = self.pos_encoder(x)              # add positional encoding
        x = self.transformer(x)              # [batch, seq, hidden_dim]
        x = x[:, -1, :]                      # use the final time step
        return self.fc(x)                    # [batch, output_dim]


# Gated Recurrent Units
class SleepStageGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SleepStageGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

# Recurrent Neural Networks
class SleepStageRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SleepStageRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out


# Long Short Term Memory
class SleepStageLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SleepStageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out