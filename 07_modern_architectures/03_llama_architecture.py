"""
=============================================================
MODÜL 7.3 — LLaMA MİMARİSİ
=============================================================

LLaMA (Touvron et al., 2023) — Meta AI
"Large Language Models" için açık ağırlıklı model ailesi.

LLaMA, GPT-2'den bu değişikliklerle ayrışır:
  1. RMSNorm yerine LayerNorm (Pre-LN kalıyor)
  2. RoPE positional encoding (Sinüzoidal/Learned yerine)
  3. SwiGLU activation (GELU yerine)
  4. Grouped Query Attention (LLaMA-2 70B ve sonrası)

LLaMA-3 (2024) ek değişiklikler:
  5. Daha büyük vocab (128K)
  6. Tiktoken tokenizer (BPE)
  7. Daha geniş context (8K default, 128K ile fine-tune)

Bu dosya: LLaMA bloğunu sıfırdan PyTorch ile.
=============================================================
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False


def llama_vs_gpt2():
    print("=" * 65)
    print("LLaMA vs GPT-2 MİMARİ FARKLILIKLARI")
    print("=" * 65)

    print("""
  ┌─────────────────────┬─────────────────┬──────────────────────────┐
  │ Bileşen             │ GPT-2           │ LLaMA / LLaMA-2 / 3      │
  ├─────────────────────┼─────────────────┼──────────────────────────┤
  │ Normalizasyon       │ LayerNorm       │ RMSNorm                  │
  │                     │ Post-LN→Pre-LN  │ Pre-LN                   │
  ├─────────────────────┼─────────────────┼──────────────────────────┤
  │ Aktivasyon          │ GELU            │ SwiGLU                   │
  ├─────────────────────┼─────────────────┼──────────────────────────┤
  │ Positional Enc.     │ Learned PE      │ RoPE                     │
  ├─────────────────────┼─────────────────┼──────────────────────────┤
  │ Attention           │ MHA             │ MHA (7B), GQA (70B+)     │
  ├─────────────────────┼─────────────────┼──────────────────────────┤
  │ FFN parametresi     │ 2 matris        │ 3 matris (SwiGLU gereği) │
  ├─────────────────────┼─────────────────┼──────────────────────────┤
  │ Bias                │ Var             │ Yok (attention & FFN)    │
  ├─────────────────────┼─────────────────┼──────────────────────────┤
  │ Vocab               │ 50,257          │ 32,000 (1/2), 128,256 (3)│
  └─────────────────────┴─────────────────┴──────────────────────────┘

  Her değişikliğin katkısı (ablasyon çalışmalarından):
    RMSNorm:  LayerNorm ile yakın, %7 daha hızlı hesap
    SwiGLU:   GELU'dan ~+1 puan NLP benchmark
    RoPE:     Learned PE'den daha iyi uzun context genellemesi
    GQA:      MHA kalitesiyle benzer, bellek/hız avantajı
    """)


# ─────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module if TORCH else object):
    """
    RMSNorm(x) = x / RMS(x) * γ
    RMS(x) = sqrt(mean(x²) + ε)

    LayerNorm'dan farkı: ortalama çıkarma (re-centering) YOK.
    Sadece yeniden ölçekleme (re-scaling).
    """
    def __init__(self, d_model, eps=1e-6):
        if TORCH:
            super().__init__()
            self.eps   = eps
            self.gamma = nn.Parameter(torch.ones(d_model))
        else:
            self.eps   = eps
            self.gamma = np.ones(d_model)

    def forward(self, x):
        if TORCH:
            rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
            return self.gamma * x / rms
        else:
            rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + self.eps)
            return self.gamma * x / rms


# ─────────────────────────────────────────────────────────────
# SwiGLU FFN
# ─────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module if TORCH else object):
    """
    LLaMA FFN:
      FFN(x) = W₂ (SiLU(W₁x) ⊙ W₃x)

    W₁, W₃: up-project  (d_model → d_ff)
    W₂:      down-project (d_ff → d_model)
    Bias YOK.

    d_ff = 4/3 * 4 * d_model  (SwiGLU 3 matris, GPT-2 2 matris)
    → LLaMA'da d_ff = int(8/3 * d_model), en yakın 256'nın katına yuvarlanır
    """
    def __init__(self, d_model, d_ff=None):
        if TORCH:
            super().__init__()
            d_ff = d_ff or int(8/3 * d_model)
            d_ff = (d_ff + 255) // 256 * 256  # 256'nın katına yuvarla
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
        else:
            self.d_ff = d_ff or int(8/3 * d_model)
            self.d_ff = (self.d_ff + 255) // 256 * 256

    def forward(self, x):
        if TORCH:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        else:
            def silu(z): return z / (1 + np.exp(-z))
            gate  = silu(x @ self.w1)
            value = x @ self.w3
            return (gate * value) @ self.w2


# ─────────────────────────────────────────────────────────────
# RoPE — bu dosyada özet, derinlemesine 08_rope_deep_dive'da
# ─────────────────────────────────────────────────────────────

def apply_rope(x, seq_len, base=10000.0):
    """
    x: (batch, n_heads, seq_len, d_k)
    RoPE uygula: her çift boyuta 2D rotasyon.
    """
    if not TORCH:
        return x

    d_k = x.shape[-1]
    assert d_k % 2 == 0
    half = d_k // 2

    # θ_i = base^(-2i/d)
    i = torch.arange(half, device=x.device).float()
    theta = 1.0 / (base ** (2 * i / d_k))                     # (half,)
    positions = torch.arange(seq_len, device=x.device).float() # (seq,)
    angles = torch.outer(positions, theta)                      # (seq, half)

    cos = angles.cos()[None, None, :, :]  # (1,1,seq,half)
    sin = angles.sin()[None, None, :, :]

    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos

    out = torch.zeros_like(x)
    out[..., 0::2] = x_rot_even
    out[..., 1::2] = x_rot_odd
    return out


# ─────────────────────────────────────────────────────────────
# Grouped Query Attention (GQA)
# ─────────────────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module if TORCH else object):
    """
    GQA: n_heads query kafası, n_kv_heads key/value kafası.
    Her query grubu aynı K, V'yi paylaşır.

    LLaMA-2 70B: n_heads=64, n_kv_heads=8  (8 grup, her grupta 8 query)
    LLaMA-3 8B:  n_heads=32, n_kv_heads=8
    """
    def __init__(self, d_model, n_heads, n_kv_heads):
        if TORCH:
            super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads   # her KV grubuna kaç Q kafası
        self.d_k        = d_model // n_heads

        if TORCH:
            self.Wq = nn.Linear(d_model, n_heads    * self.d_k, bias=False)
            self.Wk = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
            self.Wv = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
            self.Wo = nn.Linear(n_heads  * self.d_k, d_model,    bias=False)

    def forward(self, x, use_rope=True):
        if not TORCH:
            return x
        B, T, _ = x.shape

        q = self.Wq(x).view(B, T, self.n_heads,    self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        # RoPE uygula
        if use_rope:
            q = apply_rope(q, T)
            k = apply_rope(k, T)

        # KV'yi n_rep kez tekrarla (GQA → MHA gibi davranış)
        # k: (B, n_kv, T, d_k) → (B, n_heads, T, d_k)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Flash Attention (PyTorch 2.0+)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.Wo(out)


# ─────────────────────────────────────────────────────────────
# TAM LLAMA BLOĞU
# ─────────────────────────────────────────────────────────────

class LLaMABlock(nn.Module if TORCH else object):
    """
    LLaMA decoder bloğu:
      x = x + GQA_Attn(RMSNorm(x))      # Pre-RMSNorm self-attention
      x = x + SwiGLU_FFN(RMSNorm(x))    # Pre-RMSNorm FFN
    """
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff=None):
        if TORCH:
            super().__init__()
            self.ln1  = RMSNorm(d_model)
            self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
            self.ln2  = RMSNorm(d_model)
            self.ffn  = SwiGLUFFN(d_model, d_ff)

    def forward(self, x):
        if not TORCH:
            return x
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


def llama_demo():
    if not TORCH:
        print("PyTorch gerekli.")
        return

    print("\n" + "=" * 65)
    print("LLaMA BLOĞU DEMO")
    print("=" * 65)

    torch.manual_seed(42)

    configs = {
        "LLaMA-3 8B":  {"d": 4096, "h": 32, "kv": 8,  "L": 32, "d_ff": 14336},
        "LLaMA-3 70B": {"d": 8192, "h": 64, "kv": 8,  "L": 80, "d_ff": 28672},
    }

    for name, cfg in configs.items():
        L, d, h, kv = cfg["L"], cfg["d"], cfg["h"], cfg["kv"]
        d_ff = cfg["d_ff"]

        blok = LLaMABlock(d, h, kv, d_ff)
        per_block = sum(p.numel() for p in blok.parameters())
        total = L * per_block

        print(f"\n{name}:")
        print(f"  d={d}, n_heads={h}, n_kv_heads={kv}, d_ff={d_ff}")
        print(f"  Blok başına:  {per_block:,}")
        print(f"  {L} blok toplam: {total:,}  (~{total/1e9:.1f}B)")

    # Küçük model forward pass
    small = LLaMABlock(d_model=512, n_heads=8, n_kv_heads=2, d_ff=1408)
    x = torch.randn(2, 10, 512)
    out = small(x)
    print(f"\nKüçük LLaMA bloğu: {x.shape} → {out.shape}")


def llama_parametre_dagilimi():
    print("\n" + "=" * 65)
    print("LLaMA-3 8B PARAMETRE DAĞILIMI")
    print("=" * 65)

    # LLaMA-3 8B konfigürasyonu
    vocab    = 128256
    d        = 4096
    n_heads  = 32
    n_kv     = 8
    d_ff     = 14336
    L        = 32

    d_k = d // n_heads

    # Bileşenler
    embed_params = vocab * d

    # Attention (bias yok)
    Wq = n_heads * d_k * d
    Wk = n_kv   * d_k * d
    Wv = n_kv   * d_k * d
    Wo = n_heads * d_k * d
    attn_params = Wq + Wk + Wv + Wo

    # FFN (SwiGLU, 3 matris, bias yok)
    W1 = d * d_ff
    W2 = d_ff * d
    W3 = d * d_ff
    ffn_params = W1 + W2 + W3

    # RMSNorm (sadece gamma, bias yok)
    rms_params = 2 * d   # ln1, ln2

    per_layer = attn_params + ffn_params + rms_params
    total = embed_params + L * per_layer + d  # final rms norm

    print(f"{'Bileşen':28s}  {'Her Katman':>14}  {'Toplam ({L} katman)':>18}")
    print("-" * 65)

    rows = [
        ("Embedding",           embed_params, "—"),
        ("Attention (GQA)",     attn_params,  L * attn_params),
        ("  Wq (32 kafa)",      Wq,           L * Wq),
        ("  Wk (8 kv kafa)",    Wk,           L * Wk),
        ("  Wv (8 kv kafa)",    Wv,           L * Wv),
        ("  Wo",                Wo,           L * Wo),
        ("FFN (SwiGLU)",        ffn_params,   L * ffn_params),
        ("  W1 (gate)",         W1,           L * W1),
        ("  W3 (value)",        W3,           L * W3),
        ("  W2 (down)",         W2,           L * W2),
        ("RMSNorm x2",          rms_params,   L * rms_params),
    ]

    for name, per, tot in rows:
        if isinstance(tot, str):
            print(f"  {name:26s}  {'—':>14}  {per:>18,}")
        else:
            print(f"  {name:26s}  {per:>14,}  {tot:>18,}")

    print("-" * 65)
    print(f"  {'TOPLAM':26s}  {'—':>14}  {total:>18,}  (~{total/1e9:.1f}B)")


if __name__ == "__main__":
    llama_vs_gpt2()
    llama_demo()
    llama_parametre_dagilimi()
