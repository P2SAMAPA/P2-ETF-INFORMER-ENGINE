import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def _prob_qk(self, Q, K, sample_k, n_top):
        B, H, L, D = Q.shape
        _, _, S, _ = K.shape
        sample_k = min(sample_k, S)
        n_top = min(n_top, L)
        if sample_k <= 0 or n_top <= 0:
            return Q, torch.arange(L, device=Q.device).unsqueeze(0).unsqueeze(0).expand(B, H, -1)
        index_sample = torch.randint(0, S, (sample_k,), device=Q.device)
        K_sample = K[:, :, index_sample, :]
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1))
        M = Q_K_sample.max(dim=-1)[0] - Q_K_sample.mean(dim=-1)
        M_top = M.topk(n_top, dim=-1)[1]
        Q_reduce = Q.gather(-2, M_top.unsqueeze(-1).expand(-1, -1, -1, D))
        return Q_reduce, M_top

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape

        Q = self.q_linear(queries).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        K = self.k_linear(keys).view(B, S, self.n_heads, self.d_k).transpose(1,2)
        V = self.v_linear(values).view(B, S, self.n_heads, self.d_k).transpose(1,2)

        sample_k = min(self.factor * int(math.log(L)), S)
        n_top = min(self.factor * int(math.log(L)), L)
        if n_top < 1:
            n_top = 1
        Q_reduce, M_top = self._prob_qk(Q, K, sample_k, n_top)

        attn = torch.matmul(Q_reduce, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, V)
        zeros = torch.zeros(B, self.n_heads, L, self.d_k, device=queries.device)
        zeros.scatter_(-2, M_top.unsqueeze(-1).expand(-1, -1, -1, self.d_k), context)
        out = zeros.transpose(1,2).contiguous().view(B, L, -1)
        return self.out(out)

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.downConv = nn.Conv1d(c_in, c_in, 3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x.permute(0,2,1)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, factor):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class Encoder(nn.Module):
    def __init__(self, layers, conv_layers=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers else None

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.conv_layers and i < len(self.conv_layers):
                x = self.conv_layers[i](x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, factor):
        super().__init__()
        self.self_attn = ProbSparseAttention(d_model, n_heads, factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        attn = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn))
        attn2, _ = self.cross_attn(x, enc_out, enc_out)
        x = self.norm2(x + self.dropout(attn2))
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x

class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return x

class InformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['seq_len']
        self.label_len = config['label_len']
        self.pred_len = config['pred_len']
        self.enc_in = config['enc_in']
        self.dec_in = config['dec_in']
        self.d_model = config['d_model']
        self.dropout = config['dropout']

        self.enc_embed = nn.Linear(self.enc_in, self.d_model)
        self.dec_embed = nn.Linear(self.dec_in, self.d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.seq_len + self.label_len + self.pred_len, self.d_model))

        encoder_layers = [EncoderLayer(self.d_model, config['n_heads'], config['d_ff'], self.dropout, config['factor']) for _ in range(config['e_layers'])]
        conv_layers = [ConvLayer(self.d_model) for _ in range(config['e_layers']-1)] if config['distil'] else None
        self.encoder = Encoder(encoder_layers, conv_layers)

        decoder_layers = [DecoderLayer(self.d_model, config['n_heads'], config['d_ff'], self.dropout, config['factor']) for _ in range(config['d_layers'])]
        self.decoder = Decoder(decoder_layers)

        # Output: mu and log_sigma
        self.projection = nn.Linear(self.d_model, 2)

    def forward(self, x_enc, x_dec):
        B = x_enc.shape[0]
        enc_out = self.enc_embed(x_enc) + self.pos_enc[:, :self.seq_len, :]
        enc_out = self.encoder(enc_out)

        dec_out = self.dec_embed(x_dec) + self.pos_enc[:, :x_dec.shape[1], :]
        dec_out = self.decoder(dec_out, enc_out)

        out = self.projection(dec_out)[:, -self.pred_len:, :]  # (B, pred_len, 2)
        mu = out[:, :, 0].squeeze(-1)
        log_sigma = out[:, :, 1].squeeze(-1)
        return mu, log_sigma
