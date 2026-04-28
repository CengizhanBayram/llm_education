"""
=============================================================
MODÜL 4.2 — TRANSFORMER ENCODER BLOĞU
=============================================================

Encoder (BERT, T5 encoder kullanır):
  - Tüm tokenlar birbirini görebilir (çift yönlü attention)
  - Görev: girdi temsili öğrenmek

Bir encoder bloğu:
  x' = x + MHA(LN(x), LN(x), LN(x))        # Self-attention + residual
  x'' = x' + FFN(LN(x'))                    # FFN + residual

(Pre-LN versiyonu)

Konular:
  1. Encoder bloğu mimarisi
  2. Residual connection neden önemli?
  3. NumPy ile encoder bloğu
  4. PyTorch encoder bloğu
=============================================================
"""

import numpy as np

# Önceki modüllerden bileşenler (inline)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.gamma = np.ones(d)
        self.beta  = np.zeros(d)
        self.eps = eps

    def __call__(self, x):
        mu  = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mu) / np.sqrt(var + self.eps) + self.beta

class MHA:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        s = 1.0 / np.sqrt(d_model)
        self.Wq = np.random.randn(d_model, d_model) * s
        self.Wk = np.random.randn(d_model, d_model) * s
        self.Wv = np.random.randn(d_model, d_model) * s
        self.Wo = np.random.randn(d_model, d_model) * s

    def __call__(self, Q, K, V, mask=None):
        B, T, _ = Q.shape
        h, dk = self.n_heads, self.d_k

        def proj_heads(x, W):
            return (x @ W).reshape(B, T, h, dk).transpose(0, 2, 1, 3)

        q = proj_heads(Q, self.Wq)
        k = proj_heads(K, self.Wk)
        v = proj_heads(V, self.Wv)

        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk)
        if mask is not None:
            scores = np.where(mask[None, None], scores, -1e9)

        attn = softmax(scores)
        ctx  = attn @ v
        ctx  = ctx.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        return ctx @ self.Wo

class FFN:
    def __init__(self, d_model, d_ff=None):
        d_ff = d_ff or 4 * d_model
        s = np.sqrt(2.0 / (d_model + d_ff))
        self.W1 = np.random.randn(d_model, d_ff) * s
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * s
        self.b2 = np.zeros(d_model)

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def __call__(self, x):
        return self.gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


# ─────────────────────────────────────────────────────────────
# 1. ENCODER BLOĞU
# ─────────────────────────────────────────────────────────────
# Pre-LN Encoder:
#   x = x + MHA(LN(x))       [self-attention + residual]
#   x = x + FFN(LN(x))       [feed-forward + residual]
#
# İki alt katman, her biri residual connection ile sarılmış.

class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
        self.d_model = d_model
        self.mha = MHA(d_model, n_heads)
        self.ffn = FFN(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.dropout = dropout

    def __call__(self, x, mask=None):
        """
        x: (batch, seq, d_model)
        mask: (seq, seq) — padding mask veya None
        """
        # Alt katman 1: Self-attention + residual (Pre-LN)
        x_norm = self.ln1(x)
        attn_out = self.mha(x_norm, x_norm, x_norm, mask=mask)
        x = x + attn_out   # RESIDUAL CONNECTION

        # Alt katman 2: FFN + residual (Pre-LN)
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out    # RESIDUAL CONNECTION

        return x

    def param_count(self):
        mha_p = 4 * self.d_model * self.d_model
        ffn_p = 2 * self.d_model * (4 * self.d_model)
        ln_p  = 2 * 2 * self.d_model
        return mha_p + ffn_p + ln_p


def encoder_demo():
    print("=" * 60)
    print("ENCODER BLOĞU DEMO")
    print("=" * 60)

    np.random.seed(42)
    d_model, n_heads, batch, seq = 512, 8, 2, 10
    block = EncoderBlock(d_model, n_heads)

    print(f"Encoder bloğu: d={d_model}, h={n_heads}")
    print(f"Parametre sayısı: {block.param_count():,}")

    x = np.random.randn(batch, seq, d_model)
    out = block(x)
    print(f"\nGiriş: {x.shape} → Çıktı: {out.shape}")
    print(f"Giriş norm (ortalama): {np.linalg.norm(x, axis=-1).mean():.4f}")
    print(f"Çıktı norm (ortalama): {np.linalg.norm(out, axis=-1).mean():.4f}")


# ─────────────────────────────────────────────────────────────
# 2. RESİDUAL CONNECTION NEDENİ ÖNEMLİ?
# ─────────────────────────────────────────────────────────────
# He et al. 2016 "Deep Residual Learning for Image Recognition"
#
# Residual: x' = x + F(x)
# Gradyan: ∂L/∂x = ∂L/∂x' * (1 + ∂F/∂x)
#                = ∂L/∂x' + ∂L/∂x' * ∂F/∂x
#
# → Gradyan direkt akar (highway): ∂L/∂x' terimi her zaman var!
# → Derinleşince bile gradyan sıfıra gitmiyor
# → F(x) sadece artım öğrenir → başlangıçta identity yaklaşımı
#
# Görsel:
#   x ──────────────────────┐
#   │                       │
#   └─→ F(x) ─────────────→ + → x'
#
# LLM'deki Residual Stream:
#   Her katman, önceki temsili birikimli olarak günceller.
#   "Residual stream" = toplam bilgi akışı kanalı

def residual_onem():
    print("\n" + "=" * 60)
    print("2. RESİDUAL CONNECTION — NEDEN ÖNEMLİ?")
    print("=" * 60)

    print("""
  Problem: 50 katmanlı ağda, her katman küçük gradyanlar üretirse
           50 küçük sayının çarpımı → VANISHING GRADIENT

  Residual formülü: x' = x + F(x)
  Gradyan: ∂L/∂x = ∂L/∂x' * (1 + ∂F/∂x) = ∂L/∂x' + ∂L/∂x' * ∂F/∂x

  İlk terim (∂L/∂x') sadece 1 ile çarpılıyor → direkt akar!
  → Çok derin ağlarda bile gradyan kaybolmuyor.
    """)

    # Sayısal gösterim
    L = 50  # katman sayısı
    print(f"50 katmanlı ağ gradyan simülasyonu:")

    # Residual olmadan
    grad_no_res = 1.0
    for _ in range(L):
        grad_no_res *= 0.9  # her katman %10 gradyan kaybı
    print(f"  Residual YOK: {L} katman sonra gradyan = {grad_no_res:.6f}  (0.9^50)")

    # Residual ile
    grad_res = 1.0
    for _ in range(L):
        grad_res = grad_res * (1 + 0.1)  # (1 + small) ile çarpılıyor
    print(f"  Residual VAR: {L} katman sonra gradyan = {grad_res:.4f}  (büyüyor!)")

    print("\n  LLM'de residual stream:")
    print("  h_L = h_0 + Σ_{l=1}^{L} Δh_l")
    print("  Her katman küçük bir artım ekliyor, toplam öğreniliyor.")


# ─────────────────────────────────────────────────────────────
# 3. BERT ENCODERı — ÖZET
# ─────────────────────────────────────────────────────────────

def bert_encoder_ozet():
    print("\n" + "=" * 60)
    print("3. BERT ENCODER MİMARİSİ ÖZETİ")
    print("=" * 60)

    configs = {
        "BERT-base":  {"L": 12, "d": 768,  "h": 12},
        "BERT-large": {"L": 24, "d": 1024, "h": 16},
    }

    for name, cfg in configs.items():
        L, d, h = cfg["L"], cfg["d"], cfg["h"]
        block = EncoderBlock(d, h)
        per_block = block.param_count()
        embed_params = 30522 * d   # WordPiece vocab
        total = L * per_block + embed_params

        print(f"\n{name}:")
        print(f"  L={L} katman, d={d}, h={h}")
        print(f"  Blok başına: {per_block:,}")
        print(f"  {L} blok toplam: {L*per_block:,}")
        print(f"  Embedding: {embed_params:,}")
        print(f"  Toplam: ~{total/1e6:.0f}M parametre")


# ─────────────────────────────────────────────────────────────
# 4. PYTORCH ENCODER BLOĞU
# ─────────────────────────────────────────────────────────────

def pytorch_encoder():
    print("\n" + "=" * 60)
    print("4. PYTORCH ENCODER BLOĞU")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(0)

        class TransformerEncoderBlock(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
                super().__init__()
                self.attn = nn.MultiheadAttention(d_model, n_heads,
                                                   dropout=dropout, batch_first=True)
                self.ffn  = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )
                self.ln1  = nn.LayerNorm(d_model)
                self.ln2  = nn.LayerNorm(d_model)
                self.drop = nn.Dropout(dropout)

            def forward(self, x, src_key_padding_mask=None):
                # Pre-LN Self-attention
                x_norm = self.ln1(x)
                attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                         key_padding_mask=src_key_padding_mask)
                x = x + self.drop(attn_out)

                # Pre-LN FFN
                x = x + self.drop(self.ffn(self.ln2(x)))
                return x

        d_model, n_heads, d_ff = 512, 8, 2048
        block = TransformerEncoderBlock(d_model, n_heads, d_ff)
        params = sum(p.numel() for p in block.parameters())
        print(f"Encoder Bloğu: {params:,} parametre")

        x = torch.randn(2, 15, d_model)
        out = block(x)
        print(f"Giriş: {x.shape} → Çıktı: {out.shape}")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    encoder_demo()
    residual_onem()
    bert_encoder_ozet()
    pytorch_encoder()
