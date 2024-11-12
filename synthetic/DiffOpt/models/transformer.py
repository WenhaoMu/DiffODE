import torch
import torch.nn as nn


import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim=72, index_dim=1, hidden_dim=128, nhead=4, num_layers=2, dropout=0.1, max_len=100):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_embedding = nn.Linear(2, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim))  

        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, input, t):
        # input: (batch_size, length)
        # t: (batch_size, 1)

        batch_size, length = input.size()
        t = t.unsqueeze(-1).expand(batch_size, length).unsqueeze(-1).float()  # (batch_size, length, 1)
        
        h = torch.cat([input.unsqueeze(-1), t], dim=2)  # (batch_size, length, 2)
        h = self.input_embedding(h)  # (batch_size, length, hidden_dim)

        h = h + self.positional_encoding[:, :length, :]  # (batch_size, length, hidden_dim)

        h = h.permute(1, 0, 2)  # (length, batch_size, hidden_dim)

        output = self.transformer_encoder(h)  # (length, batch_size, hidden_dim)

        output = self.output_layer(output).permute(1, 0, 2)  # (batch_size, length, 1)
        return output.squeeze(-1)  # (batch_size, length)
