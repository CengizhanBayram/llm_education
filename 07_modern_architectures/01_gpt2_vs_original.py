"""
=============================================================
MODÜL 7.1 — GPT-2 vs ORİJİNAL TRANSFORMER
=============================================================

GPT-2 (Radford et al., 2019) orijinal transformerdan farklı.
Bu farklılıklar artık modern LLM'lerin standartları.

Farklar:
  1. Post-LN → Pre-LN
  2. Weight Initialization değişimi
  3. Byte-level BPE tokenizer
  4. Daha büyük context (1024 token)
  5. Tied token & position embeddings fikrinin gelişimi

Ayrıca: GPT-3, GPT-NeoX ve GPT-J ile karşılaştırma.
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


def mimari_farkliliklari():
    print("=" * 65)
    print("GPT-2 vs ORİJİNAL TRANSFORMER FARKLILIKLARI")
    print("=" * 65)

    print("""
  ┌──────────────────────┬──────────────────────┬──────────────────────┐
  │ Özellik              │ Orig. Transformer    │ GPT-2                │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ LN Pozisyonu         │ Post-LN              │ Pre-LN               │
  │                      │ x' = LN(x + F(x))   │ x' = x + F(LN(x))   │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ Son LN               │ Yok                  │ Var (final LN)       │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ Dropout              │ Her yerde 0.1        │ Azaltılmış           │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ Tokenizer            │ WordPiece / BPE      │ Byte-level BPE       │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ Context Length       │ 512                  │ 1024                 │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ Weight Tying         │ Yok                  │ tok_emb = lm_head    │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ Positional Emb.      │ Sinüzoidal (sabit)   │ Learned              │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ Init (residual)      │ N(0, 0.02)           │ N(0, 0.02/√(2L))     │
  └──────────────────────┴──────────────────────┴──────────────────────┘
    """)


def preln_faydasi():
    print("\n" + "=" * 65)
    print("PRE-LN FAYDASI — SAYISAL GÖSTERIM")
    print("=" * 65)

    np.random.seed(42)
    n_layers = 24
    d        = 512

    # Post-LN: x = LN(x + F(x))
    # Her katmanda LN uygulandıktan SONRA residual → gradyan sinyal zayıflıyor
    # Pre-LN:  x = x + F(LN(x))
    # Residual path temiz → gradyan her zaman akar

    def simulate_forward(n_layers, pre_ln=True, noise_std=0.1):
        x = np.random.randn(1, 10, d) * 0.1
        norms = []
        for l in range(n_layers):
            W = np.random.randn(d, d) * (0.02 / np.sqrt(2 * n_layers))
            if pre_ln:
                # Pre-LN: normalize sonra dönüştür, sonra ekle
                x_n = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-5)
                x   = x + x_n @ W * noise_std
            else:
                # Post-LN: dönüştür, ekle, sonra normalize
                x_new = x + x @ W * noise_std
                mu  = x_new.mean(-1, keepdims=True)
                std = x_new.std(-1, keepdims=True)
                x   = (x_new - mu) / (std + 1e-5)
            norms.append(float(np.linalg.norm(x)))
        return norms

    norms_pre  = simulate_forward(n_layers, pre_ln=True)
    norms_post = simulate_forward(n_layers, pre_ln=False)

    print(f"{'Katman':>8}  {'Pre-LN norm':>14}  {'Post-LN norm':>14}")
    print("-" * 40)
    for l in [0, 5, 11, 17, 23]:
        print(f"{l+1:>8}  {norms_pre[l]:>14.4f}  {norms_post[l]:>14.4f}")

    print(f"\nPre-LN final std:  {np.std(norms_pre):.4f}   (daha kararlı)")
    print(f"Post-LN final std: {np.std(norms_post):.4f}   (daha değişken)")


def weight_tying_aciklama():
    print("\n" + "=" * 65)
    print("WEIGHT TYING (Ağırlık Paylaşımı)")
    print("=" * 65)

    print("""
  Fikir: LM head = Token Embedding^T
    Token Embedding E ∈ R^{V × d}
    LM Head         W ∈ R^{d × V} = E^T

  Neden mantıklı?
    - Embedding: token ID → anlam vektörü
    - LM Head:  anlam vektörü → hangi token?
    - Bu iki işlem "ters" → aynı matrisle yapılabilir

  Parametre tasarrufu:
    vocab=50257, d=768 → 38.6M parametre tasarruf
    GPT-2 small toplam 117M → %33 tasarruf!

  Dezavantaj:
    - Embedding ve output öğrenme hedefleri farklı olabilir
    - Büyük modellerde (GPT-3 gibi) kullanılmıyor
    """)

    if TORCH:
        vocab, d = 50257, 768
        tok_emb = nn.Embedding(vocab, d)
        lm_head = nn.Linear(d, vocab, bias=False)

        # Weight tying
        lm_head.weight = tok_emb.weight

        tied_params   = sum(p.numel() for p in set([tok_emb.weight, lm_head.weight]))
        normal_params = tok_emb.weight.numel() + lm_head.weight.numel()
        print(f"  Normal (ayrı): {normal_params:,} parametre")
        print(f"  Tied (paylaşımlı): {tied_params:,} parametre  (-{(normal_params-tied_params)/1e6:.1f}M)")


def gpt_ailesi_karsilastirmasi():
    print("\n" + "=" * 65)
    print("GPT AİLESİ MİMARİ KARŞILAŞTIRMASI")
    print("=" * 65)

    models = [
        # name,         L,   d,    h,   d_ff,  ctx,   vocab,   PE
        ("GPT-1",        12,  768,  12,  3072,  512,   40478,   "Sinüzoidal"),
        ("GPT-2 small",  12,  768,  12,  3072,  1024,  50257,   "Learned"),
        ("GPT-2 XL",     48,  1600, 25,  6400,  1024,  50257,   "Learned"),
        ("GPT-3 175B",   96,  12288,96,  49152, 2048,  50257,   "Learned"),
        ("GPT-NeoX 20B", 44,  6144, 64,  24576, 2048,  50432,   "RoPE"),
    ]

    print(f"{'Model':>16}  {'L':>4} {'d':>6} {'h':>4} {'d_ff':>6} {'ctx':>6}  {'PE':>12}")
    print("-" * 65)
    for name, L, d, h, d_ff, ctx, vocab, pe in models:
        total = L * (4*d*d + 2*d*d_ff) + vocab*d
        print(f"{name:>16}  {L:>4} {d:>6} {h:>4} {d_ff:>6} {ctx:>6}  {pe:>12}  (~{total/1e9:.0f}B)")


def pytorch_gpt2_blok():
    if not TORCH:
        return

    print("\n" + "=" * 65)
    print("PYTORCH GPT-2 BLOĞU (Tam, Pre-LN)")
    print("=" * 65)

    class GPT2Block(nn.Module):
        """GPT-2 orijinal bloğu: Pre-LN + GELU + Weight Tying."""
        def __init__(self, d_model=768, n_heads=12, d_ff=3072, dropout=0.1):
            super().__init__()
            self.ln_1  = nn.LayerNorm(d_model)
            self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
            self.ln_2  = nn.LayerNorm(d_model)
            self.mlp   = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),              # GPT-2'de approximate GELU
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            # Residual projeksiyonlarını 1/√(2L) ile ölçekle
            # (GPT-2'de bu init time'da yapılır, burada dummy)

        def forward(self, x):
            T = x.shape[1]
            # Causal mask (üst üçgen -inf)
            mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
            # Pre-LN self-attention
            h, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=mask)
            x = x + h
            # Pre-LN FFN
            x = x + self.mlp(self.ln_2(x))
            return x

    torch.manual_seed(0)
    blok = GPT2Block()
    params = sum(p.numel() for p in blok.parameters())
    x = torch.randn(2, 10, 768)
    out = blok(x)
    print(f"GPT-2 Bloğu: {params:,} parametre")
    print(f"  Giriş: {x.shape}  →  Çıktı: {out.shape}")


if __name__ == "__main__":
    mimari_farkliliklari()
    preln_faydasi()
    weight_tying_aciklama()
    gpt_ailesi_karsilastirmasi()
    pytorch_gpt2_blok()
