import torch
import torch.nn as nn


class MAIEncoder(nn.Module):
    """One MAIFormer encoder layer (pre-LayerNorm): MMA -> AA -> FFN.

    Input tokens are laid out agent by agent, n_vars tokens per agent:
        x : (B, N * n_vars, d_model)   e.g. [lat_1, lon_1, alt_1, lat_2, lon_2, alt_2, ...]

    MMA  Masked Multivariate Attention: each token attends only to the tokens of
         its own agent (block-diagonal mask).
    AA   Agent Attention: the n_vars tokens of each agent are concatenated into one
         agent token of size n_vars * d_model, and agents attend to each other.
    FFN  position-wise feed-forward on every token.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, n_vars=3):
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars

        self.mma = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.aa = nn.MultiheadAttention(d_model * n_vars, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)              # before MMA
        self.norm2 = nn.LayerNorm(d_model * n_vars)     # before AA (on the concatenated agent token)
        self.norm3 = nn.LayerNorm(d_model)              # before FFN
        self.dropout = nn.Dropout(dropout)

    def mma_mask(self, num_agents, device):
        """(N * n_vars, N * n_vars) bool mask, True = blocked.
        A token may only attend to tokens of the same agent."""
        agent_id = torch.arange(num_agents * self.n_vars, device=device) // self.n_vars
        return agent_id.unsqueeze(0) != agent_id.unsqueeze(1)

    def forward(self, x, agent_mask=None, return_attn=False):
        """
        x          : (B, N * n_vars, d_model)
        agent_mask : (B, N) bool, True = real agent, False = padding (optional)
        returns    : x (same shape), AA weights (B, N, N), MMA weights (B, N*n_vars, N*n_vars)
                     (weights are None unless return_attn=True)
        """
        B, L, D = x.shape
        N = L // self.n_vars

        # 1) MMA: attention within each agent
        h = self.norm1(x)
        h, mma_attn = self.mma(h, h, h, attn_mask=self.mma_mask(N, x.device),
                               need_weights=return_attn)
        x = x + self.dropout(h)

        # 2) AA: concat each agent's n_vars tokens -> (B, N, n_vars * D), attention across agents
        a = x.reshape(B, N, self.n_vars * D)
        h = self.norm2(a)
        pad = None if agent_mask is None else ~agent_mask
        h, aa_attn = self.aa(h, h, h, key_padding_mask=pad, need_weights=return_attn)
        a = a + self.dropout(h)
        x = a.reshape(B, L, D)

        # 3) FFN on every token
        x = x + self.dropout(self.ffn(self.norm3(x)))

        return x, aa_attn, mma_attn
