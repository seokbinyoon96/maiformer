import torch.nn as nn

from encoder import MAIEncoder


class MAIFormer(nn.Module):
    """Multi-Agent Inverted Transformer for multi-aircraft trajectory prediction.

    Inverted embedding: each variate's whole past series (length seq_len) becomes one
    token via Linear(seq_len -> d_model). The encoder stack mixes variates within an
    aircraft (MMA) and aircraft with each other (AA). Each variate token is then
    projected to its future series with Linear(d_model -> pred_len).

    Input / output layout (variates of each agent stored next to each other):
        x   : (B, seq_len,  N * n_vars)   [lat_1, lon_1, alt_1, lat_2, ...]
        out : (B, pred_len, N * n_vars)
    """

    def __init__(self, seq_len, pred_len, n_vars=3, d_model=128, n_heads=8,
                 d_ff=512, n_layers=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars

        self.embedding = nn.Linear(seq_len, d_model)                 # inverted embedding
        self.encoders = nn.ModuleList([
            MAIEncoder(d_model, n_heads, d_ff, dropout, n_vars) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)                            # final norm (pre-LN stack)
        self.projector = nn.Linear(d_model, pred_len)                # d_model -> pred_len

    def forward(self, x, agent_mask=None, return_attn=False):
        """
        x          : (B, seq_len, N * n_vars)
        agent_mask : (B, N) bool, True = real agent, False = padding (optional)
        returns    : pred (B, pred_len, N * n_vars),
                     and if return_attn: lists of AA / MMA weights, one per layer
        """
        assert x.size(-1) % self.n_vars == 0, "last dim must be N * n_vars"

        h = self.embedding(x.transpose(1, 2))                         # (B, N * n_vars, d_model)

        aa_attns, mma_attns = [], []
        for encoder in self.encoders:
            h, aa_attn, mma_attn = encoder(h, agent_mask, return_attn)
            aa_attns.append(aa_attn)
            mma_attns.append(mma_attn)

        pred = self.projector(self.norm(h)).transpose(1, 2)           # (B, pred_len, N * n_vars)

        if return_attn:
            return pred, aa_attns, mma_attns
        return pred
