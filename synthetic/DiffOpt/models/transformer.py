import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim=72, index_dim=1, hidden_dim=128, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_embedding = nn.Linear(input_dim + index_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim))  

        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, input, t):

        sz = input.size()
        input = input.view(-1, self.input_dim)
        t = t.view(-1, self.index_dim).float()

        h = torch.cat([input, t], dim=1)
        h = self.input_embedding(h).unsqueeze(0) + self.positional_encoding[:, :h.size(0), :]
        print(h.shape)

        # h = h.unsqueeze(1)  # (batch_size, hidden_dim) -> (seq_len=1, batch_size, hidden_dim)

        output = self.transformer_encoder(h)

        output = self.output_layer(output.squeeze(0))  # (seq_len, batch_size, hidden_dim) -> (batch_size, input_dim)
        return output.view(*sz)
