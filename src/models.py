from typing import Optional, Tuple, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1) SEQ2SEQ + ATTENTION


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int, encoder_size: Optional[int] = None):
        super().__init__()
        encoder_size = encoder_size or hidden_size
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h = nn.Linear(encoder_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if decoder_hidden.dim() == 3:
            decoder_hidden = decoder_hidden[-1]  # (batch, hidden)

        dec = decoder_hidden.unsqueeze(1)  # (batch, 1, hidden)
        energy = torch.tanh(self.W_s(dec) + self.W_h(encoder_outputs))  # (batch, src_len, hidden)
        scores = self.v(energy).squeeze(-1)  # (batch, src_len)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, enc_dim)
        return context, attn_weights


class LuongAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        encoder_size: Optional[int] = None,
        method: Literal["dot", "general", "concat"] = "general",
    ):
        super().__init__()
        encoder_size = encoder_size or hidden_size
        self.method = method
        self.hidden_size = hidden_size

        if method == "general":
            self.W = nn.Linear(encoder_size, hidden_size, bias=False)
        elif method == "concat":
            self.W = nn.Linear(hidden_size + encoder_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if decoder_hidden.dim() == 3:
            decoder_hidden = decoder_hidden[-1]

        dec = decoder_hidden.unsqueeze(1)  # (batch, 1, hidden)

        if self.method == "dot":
            scores = torch.bmm(dec, encoder_outputs.transpose(1, 2)).squeeze(1)
        elif self.method == "general":
            transformed = self.W(encoder_outputs)
            scores = torch.bmm(dec, transformed.transpose(1, 2)).squeeze(1)
        else:  # concat
            src_len = encoder_outputs.size(1)
            dec_expanded = dec.expand(-1, src_len, -1)
            concat = torch.cat([dec_expanded, encoder_outputs], dim=-1)
            scores = self.v(torch.tanh(self.W(concat))).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        RNN = nn.LSTM if rnn_type == "lstm" else nn.GRU

        self.rnn = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_proj = nn.Linear(hidden_size * 2, hidden_size) if bidirectional else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        x = self.input_proj(src)

        if self.rnn_type == "lstm":
            outputs, (hidden, cell) = self.rnn(x)
        else:
            outputs, hidden = self.rnn(x)
            cell = None

        outputs = self.output_proj(outputs)
        outputs = self.layer_norm(outputs)

        if self.bidirectional:
            batch_size = hidden.size(1)
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size).mean(dim=1)
            if cell is not None:
                cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size).mean(dim=1)

        if self.rnn_type == "lstm":
            return outputs, (hidden, cell)
        return outputs, (hidden,)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        attention_type: Literal["bahdanau", "luong"] = "bahdanau",
        luong_method: Literal["dot", "general", "concat"] = "general",
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.output_size = output_size

        if attention_type == "bahdanau":
            self.attention = BahdanauAttention(hidden_size)
        else:
            self.attention = LuongAttention(hidden_size, method=luong_method)

        rnn_input_size = output_size + hidden_size
        RNN = nn.LSTM if rnn_type == "lstm" else nn.GRU

        self.rnn = RNN(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(
        self,
        input_step: torch.Tensor,
        hidden: torch.Tensor,
        cell: Optional[torch.Tensor],
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        context, attn_weights = self.attention(hidden, encoder_outputs, mask)
        rnn_input = torch.cat([input_step, context], dim=-1).unsqueeze(1)

        if self.rnn_type == "lstm":
            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        else:
            output, hidden = self.rnn(rnn_input, hidden)
            cell = None

        output = output.squeeze(1)
        pred = self.fc_out(torch.cat([output, context], dim=-1))
        return pred, hidden, cell, attn_weights


class Seq2SeqAttentionModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        horizon: int,
        enc_layers: int = 2,
        dec_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        attention_type: Literal["bahdanau", "luong"] = "bahdanau",
        bidirectional_encoder: bool = False,
    ):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.rnn_type = rnn_type

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=enc_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional_encoder,
        )

        self.decoder = Decoder(
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=dec_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            attention_type=attention_type,
        )

        self.hidden_adapter = nn.Linear(hidden_size, hidden_size) if enc_layers != dec_layers else None

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = src.size(0)
        device = src.device

        encoder_outputs, states = self.encoder(src)

        if self.rnn_type == "lstm":
            hidden, cell = states
        else:
            hidden = states[0]
            cell = None

        if self.hidden_adapter is not None:
            hidden = self.hidden_adapter(hidden)
            if cell is not None:
                cell = self.hidden_adapter(cell)

        outputs = torch.zeros(batch_size, self.horizon, self.output_size, device=device)
        input_step = torch.zeros(batch_size, self.output_size, device=device)

        for t in range(self.horizon):
            pred, hidden, cell, _attn = self.decoder(input_step, hidden, cell, encoder_outputs)
            outputs[:, t, :] = pred

            if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_step = tgt[:, t].unsqueeze(-1)
            else:
                input_step = pred

        return outputs.squeeze(-1)

    def get_attention_weights(self, src: torch.Tensor) -> torch.Tensor:
        batch_size = src.size(0)
        device = src.device

        encoder_outputs, states = self.encoder(src)

        if self.rnn_type == "lstm":
            hidden, cell = states
        else:
            hidden = states[0]
            cell = None

        all_attn = []
        input_step = torch.zeros(batch_size, self.output_size, device=device)

        for _t in range(self.horizon):
            pred, hidden, cell, attn = self.decoder(input_step, hidden, cell, encoder_outputs)
            all_attn.append(attn)
            input_step = pred

        return torch.stack(all_attn, dim=1)  # (batch, horizon, src_len)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 2) PATCHTST 

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.gamma + self.beta
        elif mode == "denorm":
            if self.affine:
                x = (x - self.beta) / self.gamma
            x = x * self.std + self.mean
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int, padding_patch: str = "end"):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        seq_len = x.size(1)
        if self.padding_patch == "end":
            remainder = (seq_len - self.patch_len) % self.stride
            if remainder != 0:
                padding = self.stride - remainder
                x = F.pad(x, (0, padding), mode="replicate")

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        n_patches = patches.size(1)
        patches = self.proj(patches)
        return patches, n_patches


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(context)
        return out, attn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm1(x)
        attn_out, attn = self.self_attn(x_norm, mask)
        x = x + self.dropout1(attn_out)

        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout2(ff_out)
        return x, attn


class PatchTST(nn.Module):
    def __init__(
        self,
        n_features: int,
        history_len: int,
        horizon: int,
        patch_len: int = 16,
        stride: Optional[int] = None,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_revin: bool = True,
        channel_independence: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.history_len = history_len
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride or patch_len
        self.d_model = d_model
        self.channel_independence = channel_independence

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(n_features)

        self.patch_embed = PatchEmbedding(
            patch_len=patch_len,
            stride=self.stride,
            d_model=d_model,
        )

        self.n_patches = (history_len - patch_len) // self.stride + 1
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.n_patches + 10, dropout=dropout)

        d_ff = d_ff or 4 * d_model
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        flatten_dim = self.n_patches * d_model
        if channel_independence:
            self.head = nn.Sequential(
                nn.Linear(flatten_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, horizon),
            )
            self.channel_aggregator = nn.Linear(n_features, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(flatten_dim * n_features, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, horizon),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch_size = x.size(0)

        if self.use_revin:
            x = self.revin(x, mode="norm")

        all_attn = []

        if self.channel_independence:
            x = x.permute(0, 2, 1).contiguous().view(batch_size * self.n_features, self.history_len)
            patches, _n = self.patch_embed(x)
            patches = self.pos_encoding(patches)

            z = patches
            for layer in self.encoder_layers:
                z, attn = layer(z)
                if return_attention:
                    all_attn.append(attn)

            z = self.final_norm(z)
            z = z.view(batch_size * self.n_features, -1)
            z = self.head(z)
            z = z.view(batch_size, self.n_features, self.horizon)

            out = self.channel_aggregator(z.permute(0, 2, 1)).squeeze(-1)
        else:
            raise NotImplementedError("Non channel-independent version not implemented")

        if return_attention:
            return out, all_attn
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


# ============================================================
# 3) FACTORY POUR SIMPLIFIER train.py
# ============================================================

def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Instancie un modèle à partir d'un nom.

    Exemple:
      model = create_model("patchtst", n_features=12, history_len=24, horizon=12)
      model = create_model("seq2seq", input_size=12, hidden_size=64, horizon=12)
    """
    name = model_name.lower().strip()
    if name in ("patchtst", "patch_tst", "patch-transformer"):
        return PatchTST(**kwargs)
    if name in ("seq2seq", "seq2seq_attention", "attention"):
        return Seq2SeqAttentionModel(**kwargs)
    raise ValueError(f"Unknown model_name={model_name}. Use 'patchtst' or 'seq2seq'.")


# Test rapide
if __name__ == "__main__":
    x = torch.randn(8, 24, 12)

    m1 = PatchTST(n_features=12, history_len=24, horizon=12, patch_len=4, d_model=64, n_heads=4, n_layers=2)
    y1 = m1(x)
    print("PatchTST:", y1.shape)

    m2 = Seq2SeqAttentionModel(input_size=12, hidden_size=64, horizon=12, bidirectional_encoder=True)
    y2 = m2(x)
    print("Seq2Seq:", y2.shape)
