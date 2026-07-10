import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- RMSNorm (Root Mean Square Layer Normalization) ----------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        inv_rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight

# ---------- SwiGLU (Swish‑Gated Linear Unit) ----------
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)

# ---------- Rotary Position Embedding (RoPE) ----------
def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len)
    angles = t.unsqueeze(1) * freqs.unsqueeze(0)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis

def apply_rotary_emb(x, freqs_cis):
    # x: [batch, seq, heads, head_dim]
    batch, seq_len, num_heads, head_dim = x.shape
    x_reshaped = x.reshape(batch, seq_len, num_heads, head_dim // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)
    freqs_cis = freqs_cis[:seq_len].unsqueeze(0).unsqueeze(2)
    out_complex = x_complex * freqs_cis
    out_real = torch.view_as_real(out_complex)
    return out_real.flatten(3)

# ---------- Multi‑Head Attention (with RoPE + dropout) ----------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)

        # Store dropout probability (will be passed to SDPA)
        self.dropout_prob = dropout

        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, max_seq_len), persistent=False)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.size()
        q = self.w_q(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.w_k(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.w_v(x).view(batch, seq_len, self.num_heads, self.head_dim)

        freqs_cis = self.freqs_cis[:seq_len]
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Transpose to (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Convert binary mask (1=allowed, 0=masked) to additive mask (-inf for masked)
        attn_mask = None
        if mask is not None:
            attn_mask = mask.masked_fill(mask == 0, float('-inf'))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_prob if self.training else 0.0
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.w_o(out)

# ---------- Transformer Block (with dropout on residual paths) ----------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, max_seq_len, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads, max_seq_len, dropout=dropout)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        # Dropout applied to the output of each sub‑layer before the residual addition
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre‑norm + attention + residual with dropout
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Pre‑norm + feed‑forward + residual with dropout
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

# ---------- Algorithm Scorer (Main Model) ----------
class AlgorithmScorer(nn.Module):
    """
    Encodes a algorithm (sequence of action tokens) into a scalar score.
    The model is trained so that **higher score = worse** (more equivalent steps).
    """
    def __init__(self, vocab_size, dim=128, ff_hidden_dim=512, num_layers=4, num_heads=4, max_seq_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_hidden_dim, max_seq_len, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.value_head = nn.Linear(dim, 1)   # projects pooled vector to a single scalar

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)          # [batch, seq_len, dim]
        if mask is None:
            seq_len = input_ids.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(1, 1, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)
        pooled = x.mean(dim=1)                 # mean pooling
        score = self.value_head(pooled).squeeze(-1)
        return score