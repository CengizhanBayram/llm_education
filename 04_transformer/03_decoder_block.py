"""
=============================================================
MODÜL 4.3 — TRANSFORMER DECODER BLOĞU
=============================================================

Decoder (Orijinal Transformer, GPT yaklaşımları):
  Tam Decoder (T5, seq2seq):
    - Alt katman 1: Masked Self-Attention (sadece geçmişi görür)
    - Alt katman 2: Cross-Attention (encoder çıktısına bakar)
    - Alt katman 3: FFN

  Decoder-only (GPT-2, GPT-3, LLaMA — en yaygın LLM mimarisi):
    - Alt katman 1: Masked Self-Attention
    - Alt katman 2: FFN
    (Cross-attention yok! Encoder yok!)

Konular:
  1. Decoder bloğu mimarisi (tam ve decoder-only)
  2. Masked Self-Attention — causal mask
  3. Cross-Attention (seq2seq için)
  4. GPT tarzı decoder-only
  5. PyTorch implementasyonu
=============================================================
"""

import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)

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
        B, Tq, _ = Q.shape
        Tk = K.shape[1]
        h, dk = self.n_heads, self.d_k

        def proj_heads_q(x):
            return (x @ self.Wq).reshape(B, Tq, h, dk).transpose(0, 2, 1, 3)
        def proj_heads_k(x):
            return (x @ self.Wk).reshape(B, Tk, h, dk).transpose(0, 2, 1, 3)
        def proj_heads_v(x):
            return (x @ self.Wv).reshape(B, Tk, h, dk).transpose(0, 2, 1, 3)

        q = proj_heads_q(Q)
        k = proj_heads_k(K)
        v = proj_heads_v(V)

        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk)
        if mask is not None:
            scores = np.where(mask[None, None], scores, -1e9)

        attn = softmax(scores)
        ctx  = (attn @ v).transpose(0, 2, 1, 3).reshape(B, Tq, self.d_model)
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
# 1. DECODER-ONLY BLOĞU (GPT TARZI)
# ─────────────────────────────────────────────────────────────
# x' = x + CausalMHA(LN(x))
# x'' = x' + FFN(LN(x'))
#
# "Decoder-only" denmesinin sebebi: sadece geçmişi görür (autoregressive)
# Ancak cross-attention YOK — seq2seq Transformer decoder'dan fark!

class GPTDecoderBlock:
    def __init__(self, d_model, n_heads, d_ff=None):
        self.d_model = d_model
        self.mha = MHA(d_model, n_heads)
        self.ffn = FFN(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def __call__(self, x):
        """
        x: (batch, seq, d_model)
        Causal mask otomatik oluşturulur
        """
        seq = x.shape[1]
        causal_mask = np.tril(np.ones((seq, seq), dtype=bool))

        # Alt katman 1: Masked Self-Attention
        x = x + self.mha(self.ln1(x), self.ln1(x), self.ln1(x), mask=causal_mask)

        # Alt katman 2: FFN
        x = x + self.ffn(self.ln2(x))

        return x


# ─────────────────────────────────────────────────────────────
# 2. TAM DECODER BLOĞU (T5, seq2seq tarzı)
# ─────────────────────────────────────────────────────────────
# x' = x + MaskedMHA(LN(x))             [masked self-attention]
# x'' = x' + CrossMHA(LN(x'), enc_out)  [cross-attention]
# x'''= x'' + FFN(LN(x''))              [feed-forward]

class FullDecoderBlock:
    def __init__(self, d_model, n_heads, d_ff=None):
        self.d_model = d_model
        self.self_mha  = MHA(d_model, n_heads)
        self.cross_mha = MHA(d_model, n_heads)
        self.ffn = FFN(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

    def __call__(self, x, enc_out, dec_mask=None, cross_mask=None):
        """
        x:       (batch, tgt_seq, d_model)  decoder girişi
        enc_out: (batch, src_seq, d_model)  encoder çıktısı
        dec_mask: causal mask
        cross_mask: encoder padding mask
        """
        tgt_seq = x.shape[1]
        causal_mask = np.tril(np.ones((tgt_seq, tgt_seq), dtype=bool))

        # Alt katman 1: Masked Self-Attention
        x = x + self.self_mha(self.ln1(x), self.ln1(x), self.ln1(x), mask=causal_mask)

        # Alt katman 2: Cross-Attention (Q = decoder, K,V = encoder)
        x_norm = self.ln2(x)
        x = x + self.cross_mha(x_norm, enc_out, enc_out, mask=cross_mask)

        # Alt katman 3: FFN
        x = x + self.ffn(self.ln3(x))

        return x


def decoder_demo():
    print("=" * 60)
    print("DECODER BLOĞU DEMO")
    print("=" * 60)

    np.random.seed(42)
    d_model, n_heads, batch = 512, 8, 2

    # GPT decoder-only
    gpt_block = GPTDecoderBlock(d_model, n_heads)
    x = np.random.randn(batch, 10, d_model)
    out_gpt = gpt_block(x)
    print(f"GPT Decoder-Only: {x.shape} → {out_gpt.shape}")

    # Tam decoder (T5 tarzı)
    full_block = FullDecoderBlock(d_model, n_heads)
    src_seq = 12
    enc_out = np.random.randn(batch, src_seq, d_model)
    tgt_x = np.random.randn(batch, 7, d_model)
    out_full = full_block(tgt_x, enc_out)
    print(f"Full Decoder: tgt={tgt_x.shape}, enc={enc_out.shape} → {out_full.shape}")


# ─────────────────────────────────────────────────────────────
# 3. CROSS-ATTENTION MEKANİZMASI
# ─────────────────────────────────────────────────────────────
# Encoder'dan gelen bilgiyi kullanma:
#   Q = decoder'dan  (ne arıyorum?)
#   K = encoder'dan  (encoder hangi bilgilere sahip?)
#   V = encoder'dan  (o bilgiler neler?)
#
# seq2seq: "I love Paris" → "J'aime Paris" (İngilizce → Fransızca)
# Decoder "J'aime" üretirken, "I love"'a cross-attention ile bakabilir.

def cross_attention_aciklama():
    print("\n" + "=" * 60)
    print("3. CROSS-ATTENTION MEKANİZMASI")
    print("=" * 60)

    print("""
  Çeviri görevi:
    Kaynak: "I  love  Paris"  → encoder çıktısı = K, V
            [0]  [1]   [2]

    Hedef: "J'  aime  Paris"  → decoder Q'su
            [0]  [1]   [2]

  Cross-attention:
    "J'" token'ı Q olarak "I"'ya bakabilir: Q[0] · K[0] yüksek
    "aime" token'ı Q olarak "love"'a bakabilir: Q[1] · K[1] yüksek
    "Paris" token'ı Q olarak "Paris"'e bakabilir: Q[2] · K[2] yüksek

  Bu alignment (hizalama) mekanizması:
    → Decoder hangi kaynak token'ından bilgi alacağını dinamik olarak seçer
    → Eski seq2seq modellerindeki attention alignment ile aynı prensip
    """)

    np.random.seed(0)
    d_k = 4
    # Simüle edilmiş: "I love Paris" → encoder representations
    enc_keys = np.array([
        [1.0, 0.0, 0.0, 0.0],  # "I"
        [0.0, 1.0, 0.0, 0.0],  # "love"
        [0.0, 0.0, 1.0, 0.0],  # "Paris"
    ])
    # Decoder sorguları: "J'" ve "aime"
    dec_queries = np.array([
        [0.9, 0.1, 0.0, 0.0],  # "J'" → "I"'ye benzer
        [0.1, 0.9, 0.0, 0.0],  # "aime" → "love"'a benzer
    ])

    scores = dec_queries @ enc_keys.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)

    print("Cross-attention ağırlıkları (decoder → encoder):")
    print(f"{'':10s} {'I':>8} {'love':>8} {'Paris':>8}")
    src_tokens = ["I", "love", "Paris"]
    tgt_tokens = ["J'", "aime"]
    for i, tok in enumerate(tgt_tokens):
        row = "  ".join([f"{w:.4f}" for w in weights[i]])
        print(f"  {tok:8s}: {row}")
    print("→ 'J'' en çok 'I'ya bakıyor, 'aime' en çok 'love'a bakıyor ✓")


# ─────────────────────────────────────────────────────────────
# 4. AUTOREGRESSİVE GENERATİON
# ─────────────────────────────────────────────────────────────
# GPT gibi decoder-only modeller nasıl üretim yapar?
#
# Adım 1: P(w_1 | <BOS>) → token seç
# Adım 2: P(w_2 | <BOS>, w_1) → token seç
# Adım 3: P(w_3 | <BOS>, w_1, w_2) → token seç
# ...
#
# Her adımda sadece bir token üretilir → "autoregressive"
# KV Cache: önceki K, V değerlerini cache'le → her adımda O(1) hesap

def autoregressive_demo():
    print("\n" + "=" * 60)
    print("4. AUTOREGRESSİVE ÜRETIM SİMÜLASYONU")
    print("=" * 60)

    np.random.seed(42)
    vocab = 10   # küçük vocab
    d_model, n_heads = 64, 4
    max_new_tokens = 5

    # Küçük GPT bloğu
    block = GPTDecoderBlock(d_model, n_heads)

    # Embedding (rastgele)
    token_emb = np.random.randn(vocab, d_model) * 0.1
    pos_emb   = np.random.randn(50, d_model) * 0.01
    W_out = np.random.randn(d_model, vocab) * 0.1  # LM head

    def softmax_1d(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    # Başlangıç token
    input_ids = [0]   # BOS token
    print(f"Üretim başlatılıyor... Başlangıç: {input_ids}")

    for step in range(max_new_tokens):
        seq = len(input_ids)
        # Embedding
        x = np.array([token_emb[t] + pos_emb[i] for i, t in enumerate(input_ids)])
        x = x[np.newaxis]   # (1, seq, d_model)

        # Forward pass
        out = block(x)   # (1, seq, d_model)

        # Son token için logitler
        logits = out[0, -1] @ W_out   # (vocab,)
        probs = softmax_1d(logits)

        # Greedy decode
        next_token = np.argmax(probs)
        input_ids.append(int(next_token))
        print(f"  Adım {step+1}: logits.max={logits.max():.3f}, "
              f"seçilen token={next_token} (p={probs[next_token]:.4f})")

    print(f"\nÜretilen sekans: {input_ids}")

    # KV Cache konsepti
    print("\n--- KV CACHE ─────────────────────────────────────")
    print("""
  KV Cache olmadan: her yeni token için tüm sekansı yeniden hesapla
    Adım t için: O(t) hesap
    Toplam N token için: O(N²) hesap

  KV Cache ile: önceki K, V değerlerini sakla
    Adım t için: sadece yeni token'ı işle → O(1)
    Toplam N token için: O(N) hesap

  Bellek tradeoff:
    L katman × n_heads × (seq_len × d_k) × 2 (K+V)
    GPT-3 175B: ~40GB KV cache (seq_len=2048 için)
    → GQA/MQA ile azaltılır (LLaMA-2)
    """)


def pytorch_decoder():
    print("\n" + "=" * 60)
    print("5. PYTORCH DECODER BLOĞU")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        torch.manual_seed(42)

        class GPTBlock(nn.Module):
            def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
                super().__init__()
                self.ln1  = nn.LayerNorm(d_model)
                self.attn = nn.MultiheadAttention(d_model, n_heads,
                                                   dropout=dropout, batch_first=True)
                self.ln2  = nn.LayerNorm(d_model)
                self.ffn  = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                )

            def forward(self, x):
                T = x.shape[1]
                # Causal mask
                causal = nn.Transformer.generate_square_subsequent_mask(T,
                                                                         device=x.device)
                # Masked self-attention
                x_norm = self.ln1(x)
                attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal)
                x = x + attn_out

                # FFN
                x = x + self.ffn(self.ln2(x))
                return x

        d_model, n_heads, d_ff = 512, 8, 2048
        block = GPTBlock(d_model, n_heads, d_ff)
        params = sum(p.numel() for p in block.parameters())
        print(f"GPT Decoder Bloğu: {params:,} parametre")

        x = torch.randn(2, 10, d_model)
        out = block(x)
        print(f"Giriş: {x.shape} → Çıktı: {out.shape}")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    decoder_demo()
    cross_attention_aciklama()
    autoregressive_demo()
    pytorch_decoder()
