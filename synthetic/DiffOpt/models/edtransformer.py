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
        self.target_embedding = nn.Linear(1, hidden_dim)

        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, src, t, tgt=None):
        """
        Args:
        - src: (batch_size, length_src) -> Input sequence
        - t: (batch_size, 1) -> Time information
        - tgt: (batch_size, length_tgt) -> Target sequence
        """
        batch_size, length_src = src.size()
        length_tgt = length_src if tgt is None else tgt.size(1)
        tgt = torch.zeros(batch_size, length_tgt, dtype=torch.long).float().to(src.device) if tgt is None else tgt

        t_expanded = t.unsqueeze(-1).expand(batch_size, length_src).unsqueeze(-1).float()  # (batch_size, length_src, 1)
        src_embedded = torch.cat([src.unsqueeze(-1), t_expanded], dim=2)  # (batch_size, length_src, 2)
        src_embedded = self.input_embedding(src_embedded)  # (batch_size, length_src, hidden_dim)
        src_embedded += self.positional_encoding[:, :length_src, :]
        src_embedded = src_embedded.permute(1, 0, 2)  # (length_src, batch_size, hidden_dim)

        memory = self.encoder(src_embedded)  # (length_src, batch_size, hidden_dim)

        tgt_embedded = self.target_embedding(tgt.unsqueeze(-1))  # (batch_size, length_tgt, hidden_dim)
        tgt_embedded += self.positional_encoding[:, :length_tgt, :]
        tgt_embedded = tgt_embedded.permute(1, 0, 2)  # (length_tgt, batch_size, hidden_dim)

        output = self.decoder(tgt_embedded, memory)  # (length_tgt, batch_size, hidden_dim)

        output = self.output_layer(output).permute(1, 0, 2)  # (batch_size, length_tgt, 1)
        return output.squeeze(-1)  # (batch_size, length_tgt)

