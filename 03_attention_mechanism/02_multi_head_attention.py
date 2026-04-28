"""
=============================================================
MODÜL 3.2 — MULTI-HEAD ATTENTION (MHA)
=============================================================

"Attention Is All You Need":
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O

    head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)

Neden çok kafa?
  - Tek attention: tek tip ilişkiyi öğrenir
  - Çok kafa: farklı "bakış açıları" paralel öğrenir
    → Kafa 1: sözdizimsel (syntactic) ilişkiler
    → Kafa 2: anlamsal (semantic) ilişkiler
    → Kafa 3: konum tabanlı (positional) ilişkiler
    → Kafa 4: coreference ...

Konular:
  1. MHA matematiksel formülasyon
  2. NumPy ile sıfırdan MHA
  3. Parametreler ve boyutlar
  4. Kafaların yorumlanması
  5. PyTorch MHA implementasyonu
=============================================================
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ─────────────────────────────────────────────────────────────
# 1. MATEMATİK
# ─────────────────────────────────────────────────────────────
# h kafa, her kafa d_k = d_model/h boyutlu
#
# Projeksiyon matrisleri (her kafa için):
#   W_Qi ∈ R^{d_model x d_k}   (i = 1,...,h)
#   W_Ki ∈ R^{d_model x d_k}
#   W_Vi ∈ R^{d_model x d_v}   (genellikle d_v = d_k = d_model/h)
#
# Output projeksiyon:
#   W_O ∈ R^{h*d_v x d_model}
#
# head_i = Attention(X W_Qi, X W_Ki, X W_Vi)
# MHA = Concat(head_1, ..., head_h) W_O
#
# Parametreler:
#   W_Q: d_model × d_model  (tüm kafalar birlikte)
#   W_K: d_model × d_model
#   W_V: d_model × d_model
#   W_O: d_model × d_model
#   Toplam: 4 × d_model²

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model, n_heads'in katı olmalı"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        scale = 1.0 / np.sqrt(d_model)
        # Projeksiyon matrisleri — tüm kafalar birlikte depolanır
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def split_heads(self, X):
        """
        X: (batch, seq, d_model)
        → (batch, n_heads, seq, d_k)
        """
        batch, seq, _ = X.shape
        X = X.reshape(batch, seq, self.n_heads, self.d_k)
        return X.transpose(0, 2, 1, 3)   # (batch, n_heads, seq, d_k)

    def merge_heads(self, X):
        """
        X: (batch, n_heads, seq, d_v)
        → (batch, seq, d_model)
        """
        batch, n_heads, seq, d_v = X.shape
        X = X.transpose(0, 2, 1, 3)      # (batch, seq, n_heads, d_v)
        return X.reshape(batch, seq, n_heads * d_v)

    def forward(self, Q_in, K_in, V_in, mask=None):
        """
        Q_in, K_in, V_in: (batch, seq, d_model)
        Returns: (batch, seq, d_model)
        """
        batch, seq, _ = Q_in.shape

        # 1. Lineer projeksiyon
        Q = Q_in @ self.W_Q   # (batch, seq, d_model)
        K = K_in @ self.W_K
        V = V_in @ self.W_V

        # 2. Kafaları ayır
        Q = self.split_heads(Q)   # (batch, h, seq, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. Her kafa için attention
        # scores: (batch, h, seq_q, seq_k)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            # mask: (seq_q, seq_k) → broadcast
            scores = np.where(mask[np.newaxis, np.newaxis, :, :], scores, -1e9)

        weights = softmax(scores, axis=-1)   # (batch, h, seq_q, seq_k)

        # context: (batch, h, seq_q, d_v)
        context = weights @ V

        # 4. Kafaları birleştir
        context = self.merge_heads(context)   # (batch, seq, d_model)

        # 5. Output projeksiyon
        output = context @ self.W_O           # (batch, seq, d_model)

        return output, weights

    def param_count(self):
        return 4 * self.d_model * self.d_model

    def __repr__(self):
        return (f"MultiHeadAttention("
                f"d_model={self.d_model}, n_heads={self.n_heads}, "
                f"d_k={self.d_k}, params={self.param_count():,})")


def mha_demo():
    print("=" * 60)
    print("MULTİ-HEAD ATTENTION DEMO")
    print("=" * 60)

    np.random.seed(42)
    d_model = 512
    n_heads = 8
    batch, seq = 2, 10

    mha = MultiHeadAttention(d_model, n_heads)
    print(mha)

    X = np.random.randn(batch, seq, d_model)

    # Causal mask
    mask = np.tril(np.ones((seq, seq), dtype=bool))

    output, weights = mha.forward(X, X, X, mask=mask)

    print(f"\nGiriş: {X.shape}")
    print(f"Çıktı: {output.shape}")
    print(f"Ağırlıklar: {weights.shape}  (batch, n_heads, seq, seq)")

    # Kafa başına dikkat istatistikleri
    print(f"\nKafa başına attention entropisi:")
    def entropy(p, eps=1e-9):
        return -(p * np.log(p + eps)).sum(axis=-1).mean()

    for h in range(n_heads):
        H = entropy(weights[0, h])
        print(f"  Kafa {h}: entropi = {H:.4f}")


# ─────────────────────────────────────────────────────────────
# 2. GPT-2 MİMARİSİ PARAMETRELERİ
# ─────────────────────────────────────────────────────────────

def gpt2_parametre_analizi():
    print("\n" + "=" * 60)
    print("GPT-2 MİMARİSİ PARAMETRE ANALİZİ")
    print("=" * 60)

    configs = {
        "GPT-2 small":  {"n_layers": 12, "d_model": 768,  "n_heads": 12},
        "GPT-2 medium": {"n_layers": 24, "d_model": 1024, "n_heads": 16},
        "GPT-2 large":  {"n_layers": 36, "d_model": 1280, "n_heads": 20},
        "GPT-2 XL":     {"n_layers": 48, "d_model": 1600, "n_heads": 25},
    }

    for name, cfg in configs.items():
        L = cfg["n_layers"]
        d = cfg["d_model"]
        h = cfg["n_heads"]

        d_k = d // h

        attn_params = L * 4 * d * d        # W_Q, W_K, W_V, W_O her katman
        ffn_params  = L * 2 * d * (4*d)    # FFN: d→4d→d, iki matris
        embed_params = 50257 * d            # token embedding
        pos_params   = 1024 * d             # positional embedding (GPT-2 max_len=1024)

        total = attn_params + ffn_params + embed_params + pos_params

        print(f"\n{name}:")
        print(f"  d_model={d}, n_heads={h}, d_k={d_k}, n_layers={L}")
        print(f"  Attention params:   {attn_params:>12,}  ({attn_params/total*100:.1f}%)")
        print(f"  FFN params:         {ffn_params:>12,}  ({ffn_params/total*100:.1f}%)")
        print(f"  Embedding params:   {embed_params:>12,}  ({embed_params/total*100:.1f}%)")
        print(f"  Toplam:             {total:>12,}  (~{total/1e6:.0f}M)")


# ─────────────────────────────────────────────────────────────
# 3. KAFALAR NE ÖĞRENIR?
# ─────────────────────────────────────────────────────────────
# Clark et al. (2019) "What Does BERT Look At?" bulgularından:
# - Kafa 8-10: doğrudan nesne (direct object) ilişkisi
# - Kafa 8-11: özne (subject) ilişkisi
# - Bazı kafalar: bir sonraki/önceki tokena bakıyor (positional)
# - Bazı kafalar: [SEP] tokenına yüksek ağırlık (BERT'e özgü)

def kafa_interpretability():
    print("\n" + "=" * 60)
    print("3. KAFA İNTERPRETABİLİTY SİMÜLASYONU")
    print("=" * 60)

    np.random.seed(7)
    n_heads, seq = 4, 6

    print("4 farklı kafa kalıbı simülasyonu:")
    print("Tokenlar: [The, cat, sat, on, the, mat]")
    tokens = ["The", "cat", "sat", "on", "the", "mat"]

    # Kafa 0: Diagonal (her token kendine)
    w0 = np.eye(seq)

    # Kafa 1: Bir önceki tokena bak (positional)
    w1 = np.zeros((seq, seq))
    for i in range(1, seq):
        w1[i, i-1] = 1.0
    w1[0, 0] = 1.0

    # Kafa 2: "the" → sonraki isme bak (determiner → noun)
    w2 = np.eye(seq) * 0.3
    # "The"(0) → "cat"(1), "the"(4) → "mat"(5)
    w2[0, 1] = 0.7
    w2[4, 5] = 0.7
    w2 /= w2.sum(axis=-1, keepdims=True)

    # Kafa 3: Uzak bağımlılık (sat → cat)
    w3 = np.eye(seq) * 0.2
    w3[2, 1] = 0.8  # "sat" → "cat"
    w3 /= w3.sum(axis=-1, keepdims=True)

    for kafa_idx, (name, w) in enumerate([
        ("Diagonal (kendine)",      w0),
        ("Positional (önceki)",     w1),
        ("Syntactic (det→noun)",    w2),
        ("Uzak (sat→cat)",          w3),
    ]):
        print(f"\nKafa {kafa_idx}: {name}")
        # En yüksek ağırlıklı ilişkiyi göster
        for i in range(seq):
            j = np.argmax(w[i])
            if w[i, j] > 0.4:
                print(f"  '{tokens[i]}' → '{tokens[j]}' (w={w[i,j]:.2f})")


# ─────────────────────────────────────────────────────────────
# 4. GROUPED QUERY ATTENTION (GQA) — LLaMA-2
# ─────────────────────────────────────────────────────────────
# LLaMA-2, Mistral, Gemma gibi modern LLM'ler GQA kullanır.
#
# Motivasyon:
#   - MHA: her kafa için ayrı K, V → n_heads × KV cache
#   - MQA (Multi-Query): tek K, V paylaşılır → 1 × KV cache, kalite düşük
#   - GQA (Grouped Query): g grup, her grup K,V paylaşır → g × KV cache
#
# GQA: n_heads query kafası, n_kv_heads key/value kafası
#   n_kv_heads divides n_heads
#   LLaMA-2 70B: n_heads=64, n_kv_heads=8

def gqa_demo():
    print("\n" + "=" * 60)
    print("4. GROUPED QUERY ATTENTION (GQA)")
    print("=" * 60)

    d_model = 512
    n_heads = 8
    n_kv_heads = 2   # Her 4 query kafası için 1 KV kafası
    d_k = d_model // n_heads

    # KV cache boyutu karşılaştırması
    seq_len = 2048

    mha_kv_cache = 2 * n_heads * seq_len * d_k       # K ve V, tüm kafalar
    gqa_kv_cache = 2 * n_kv_heads * seq_len * d_k    # K ve V, azaltılmış kafa
    mqa_kv_cache = 2 * 1 * seq_len * d_k             # MQA: tek kafa

    print(f"d_model={d_model}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
    print(f"seq_len={seq_len}")
    print(f"\nKV Cache boyutu:")
    print(f"  MHA (n_kv={n_heads}):     {mha_kv_cache:>10,}  (1.0x)")
    print(f"  GQA (n_kv={n_kv_heads}):      {gqa_kv_cache:>10,}  ({gqa_kv_cache/mha_kv_cache:.2f}x)")
    print(f"  MQA (n_kv=1):     {mqa_kv_cache:>10,}  ({mqa_kv_cache/mha_kv_cache:.2f}x)")
    print(f"→ GQA: kaliteden çok az ödün, bellek %{(1-gqa_kv_cache/mha_kv_cache)*100:.0f} tasarruf")


# ─────────────────────────────────────────────────────────────
# 5. PYTORCH MHA
# ─────────────────────────────────────────────────────────────

def pytorch_mha():
    print("\n" + "=" * 60)
    print("5. PYTORCH MHA")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        d_model, n_heads, seq, batch = 512, 8, 10, 2

        # PyTorch built-in
        mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        param_count = sum(p.numel() for p in mha.parameters())
        print(f"PyTorch MHA(d={d_model}, h={n_heads}): {param_count:,} parametre")

        X = torch.randn(batch, seq, d_model)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq)

        out, attn_weights = mha(X, X, X, attn_mask=causal_mask)
        print(f"Giriş: {X.shape} → Çıktı: {out.shape}")
        print(f"Attention ağırlıkları: {attn_weights.shape}")

        # Flash Attention (PyTorch 2.0+ SDPA)
        import torch.nn.functional as F

        d_k = d_model // n_heads
        # Q, K, V'yi başlık boyutuna göre böl (manuel)
        Wq = nn.Linear(d_model, d_model, bias=False)
        Wk = nn.Linear(d_model, d_model, bias=False)
        Wv = nn.Linear(d_model, d_model, bias=False)
        Wo = nn.Linear(d_model, d_model, bias=False)

        def mha_forward(X):
            B, T, C = X.shape
            q = Wq(X).view(B, T, n_heads, d_k).transpose(1, 2)
            k = Wk(X).view(B, T, n_heads, d_k).transpose(1, 2)
            v = Wv(X).view(B, T, n_heads, d_k).transpose(1, 2)
            # PyTorch 2.0 Flash Attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return Wo(y)

        with torch.no_grad():
            out2 = mha_forward(X)
        print(f"Flash Attention SDPA çıktı: {out2.shape}")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    mha_demo()
    gpt2_parametre_analizi()
    kafa_interpretability()
    gqa_demo()
    pytorch_mha()
