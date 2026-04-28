"""
=============================================================
MODÜL 3.1 — SCALED DOT-PRODUCT ATTENTION
=============================================================

"Attention Is All You Need" (Vaswani et al., 2017) makalesinin
temel formülü:

    Attention(Q, K, V) = softmax(QK^T / √d_k) V

Bu, transformer'ın kalbidir. Bu formülü derinlemesine anlaman
LLM research için kritik.

Konular:
  1. Motivasyon — neden attention?
  2. Query, Key, Value sezgisi
  3. Scaled dot-product attention — adım adım
  4. Maskeleme (Causal Mask) — GPT için kritik
  5. Attention pattern analizi
  6. PyTorch implementasyonu
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. MOTİVASYON
# ─────────────────────────────────────────────────────────────
# RNN sorunu:
#   - Bilgi sol→sağ sıralı aktarılır
#   - "The animal didn't cross the street because IT was too tired"
#     "it" → "animal" bağlantısı için tüm sekansı geçmek lazım
#   - Uzun bağımlılıklar → vanishing gradient
#
# Attention çözümü:
#   - Her token, diğer TÜM tokenlara doğrudan bakabilir
#   - Uzaklıktan bağımsız: O(1) hesaplama
#   - Hangi tokena ne kadar dikkat? → öğreniliyor!

def motivasyon():
    print("=" * 60)
    print("1. MOTİVASYON — NEDEN ATTENTION?")
    print("=" * 60)
    print("""
  Sekans: "The animal didn't cross the street because it was tired"
  Token:   0     1       2      3     4    5       6    7   8
                                                   ↑
                                            "it" = token 6

  RNN: "it" için bilgi 0→1→2→3→4→5→6 yolundan gelir → bozunabilir
  Attention: "it" doğrudan "animal" (token 1) ile ilişki kurabilir

  Attention ağırlığı a(6, 1) büyük → "it" = "animal" öğrenilir
    """)


# ─────────────────────────────────────────────────────────────
# 2. QUERY, KEY, VALUE SEZGİSİ
# ─────────────────────────────────────────────────────────────
# Veritabanı analojisi:
#   - Query (Q): "Ne arıyorum?" — mevcut token'ın sorusu
#   - Key   (K): "Ben kimim?"   — her token'ın kimliği
#   - Value (V): "Ne sunuyorum?" — her token'ın içeriği
#
# Süreç:
#   1. Q ve K'ları karşılaştır → dikkat skoru
#   2. Skoru softmax ile normalize et → dikkat ağırlığı
#   3. V'leri ağırlıklı topla → çıktı
#
# Matris dönüşümleri (öğrenilir):
#   Q = X W_Q,   W_Q ∈ R^{d_model x d_k}
#   K = X W_K,   W_K ∈ R^{d_model x d_k}
#   V = X W_V,   W_V ∈ R^{d_model x d_v}

def qkv_sezgisi():
    print("\n" + "=" * 60)
    print("2. QUERY-KEY-VALUE SEZGİSİ")
    print("=" * 60)
    print("""
  Kütüphane analojisi:
    Query: "transformerlarla ilgili kitap arıyorum"
    Keys:  kitapların etiketleri/başlıkları
    Values: kitapların içerikleri

  Adım 1: Query ile her Key benzerliğini hesapla (dot product)
  Adım 2: Benzerlikleri normalize et (softmax) → ağırlıklar
  Adım 3: Value'ları ağırlıklı topla → ilgili bilgiyi al

  Matematiksel:
    score(q, k_i) = q · k_i / √d_k
    α_i = softmax(score(q, k_i))  ∀i
    output = Σ_i α_i v_i
    """)


# ─────────────────────────────────────────────────────────────
# 3. SCALED DOT-PRODUCT ATTENTION — ADIM ADIM
# ─────────────────────────────────────────────────────────────
# Attention(Q, K, V) = softmax(QK^T / √d_k) V
#
# Boyutlar:
#   Q ∈ R^{n x d_k}  (n sorgu, d_k boyutlu)
#   K ∈ R^{m x d_k}  (m anahtar)
#   V ∈ R^{m x d_v}  (m değer, d_v boyutlu)
#   Çıktı ∈ R^{n x d_v}
#
# Neden √d_k ile ölçekle?
#   dot product = Σ_{i=1}^{d_k} q_i k_i
#   Eğer q_i, k_i ~ N(0,1) ise:
#     E[q·k] = 0,  Var(q·k) = d_k
#   → Büyük d_k → büyük varyans → softmax doyumuna girer → gradyan küçülür
#   √d_k ile böl → Var = 1 → gradyan akışı düzgün

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, n_q, d_k)
    K: (batch, n_k, d_k)
    V: (batch, n_k, d_v)
    mask: (batch, n_q, n_k) — True olan konumlar -inf yapılır

    Returns:
        output: (batch, n_q, d_v)
        weights: (batch, n_q, n_k)
    """
    d_k = Q.shape[-1]

    # Adım 1: Skorları hesapla — Q @ K^T
    # (batch, n_q, d_k) @ (batch, d_k, n_k) → (batch, n_q, n_k)
    scores = Q @ K.transpose(0, 2, 1)   # (batch, n_q, n_k)

    # Adım 2: √d_k ile ölçekle
    scores = scores / np.sqrt(d_k)

    # Adım 3: Maske uygula (opsiyonel)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # Adım 4: Softmax — (batch, n_q, n_k)
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Adım 5: Value'larla ağırlıklı toplam
    # (batch, n_q, n_k) @ (batch, n_k, d_v) → (batch, n_q, d_v)
    output = weights @ V

    return output, weights


def attention_adim_adim():
    print("\n" + "=" * 60)
    print("3. SCALED DOT-PRODUCT ATTENTION — ADIM ADIM")
    print("=" * 60)

    np.random.seed(42)
    batch, n, d_k, d_v = 1, 4, 8, 8

    # Örnek: 4 token, d_k=8
    Q = np.random.randn(batch, n, d_k)
    K = np.random.randn(batch, n, d_k)
    V = np.random.randn(batch, n, d_v)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"Girişler:")
    print(f"  Q: {Q.shape}  K: {K.shape}  V: {V.shape}")
    print(f"\nAttention ağırlıkları (batch=0):")
    np.set_printoptions(precision=4, suppress=True)
    print(weights[0])
    print(f"  Her satır toplamı: {weights[0].sum(axis=-1)}  (hepsi 1.0)")
    print(f"\nÇıktı shape: {output.shape}")

    # Ölçeklemenin önemi
    print("\n--- √d_k ÖLÇEKLEMENİN ÖNEMİ ---")
    d_büyük = 64
    q = np.random.randn(1, d_büyük)
    k = np.random.randn(1, d_büyük)

    score_olmadan = q @ k.T
    score_ile = q @ k.T / np.sqrt(d_büyük)

    def softmax_1d(x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    # Softmax öncesi skoru rastgele 10 q, 10 k için hesapla
    Q_test = np.random.randn(1, 10, d_büyük)
    K_test = np.random.randn(1, 10, d_büyük)

    scores_raw = (Q_test @ K_test.transpose(0, 2, 1))[0]
    scores_scaled = scores_raw / np.sqrt(d_büyük)

    def entropy(p):
        p = np.clip(p, 1e-9, 1)
        return -np.sum(p * np.log(p), axis=-1)

    # Entropi: düşük → doyum, yüksek → dengeli
    w_raw    = np.array([softmax_1d(scores_raw[i]) for i in range(10)])
    w_scaled = np.array([softmax_1d(scores_scaled[i]) for i in range(10)])

    print(f"d_k = {d_büyük}")
    print(f"Ham skor std:    {scores_raw.std():.4f}")
    print(f"Ölçekli skor std: {scores_scaled.std():.4f}")
    print(f"Ham attention entropy:    {entropy(w_raw).mean():.4f}")
    print(f"Ölçekli attention entropy: {entropy(w_scaled).mean():.4f}")
    print(f"→ Ham: doyum (sharp attention) → gradyan küçük")
    print(f"→ Ölçekli: dengeli attention → gradyan akışı iyi")


# ─────────────────────────────────────────────────────────────
# 4. CAUSAL MASK (NEDENSEL MASKE)
# ─────────────────────────────────────────────────────────────
# GPT (decoder-only) modeller için kritik!
# Dil modeli: w_t yalnızca w_{1:t-1}'e bakabilir, gelecek göremez.
#
# Causal mask: üçgen maske
#   mask[i, j] = True  if j <= i   (geçmiş tokenlar)
#              = False if j > i    (gelecek tokenlar → -inf → 0 ağırlık)
#
# [ T F F F ]    Token 0: sadece kendini görür
# [ T T F F ]    Token 1: 0 ve 1'i görür
# [ T T T F ]    Token 2: 0, 1, 2'yi görür
# [ T T T T ]    Token 3: hepsini görür

def causal_mask_demo():
    print("\n" + "=" * 60)
    print("4. CAUSAL MASK (NEDENSEL MASKE) — GPT")
    print("=" * 60)

    seq_len = 5

    # Alt üçgen maske
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    print("Causal mask (True = bakabilir):")
    print(mask.astype(int))

    # Batch boyutu ekle
    mask_batched = mask[np.newaxis, :, :]   # (1, seq, seq)

    np.random.seed(42)
    batch, d_k, d_v = 1, 4, 4
    Q = np.random.randn(batch, seq_len, d_k)
    K = np.random.randn(batch, seq_len, d_k)
    V = np.random.randn(batch, seq_len, d_v)

    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask_batched)
    output_full,   weights_full   = scaled_dot_product_attention(Q, K, V, mask=None)

    print("\nMaskelenmiş attention ağırlıkları:")
    print(np.round(weights_masked[0], 4))
    print("→ Üst üçgen sıfır: gelecek tokenlar görünmüyor!")

    print("\nMaskesiz attention ağırlıkları (encoder için):")
    print(np.round(weights_full[0], 4))
    print("→ Her token herkese bakabiliyor")


# ─────────────────────────────────────────────────────────────
# 5. ATTENTION PATTERN ANALİZİ
# ─────────────────────────────────────────────────────────────
# Gerçek LLM'lerde attention kalıpları:
#   - Diagonal: her token kendine yüksek ağırlık (self-attention)
#   - Syntactic: fiil → özne bağlantısı
#   - Coreference: "it" → "the animal"
#   - Positional: yakın tokenlara daha yüksek ağırlık

def attention_pattern_analizi():
    print("\n" + "=" * 60)
    print("5. ATTENTION PATTERN ANALİZİ")
    print("=" * 60)

    np.random.seed(0)
    n, d_k = 6, 16

    # Simüle edilmiş embedding: bazı tokenlar birbirine benzer
    # Token 0 ve 2 benzer (aynı konu), Token 3 ve 5 benzer
    embeddings = np.random.randn(n, d_k) * 0.5
    embeddings[0] += np.array([1.0] * d_k)   # token 0 ve 2 benzer
    embeddings[2] += np.array([1.0] * d_k)
    embeddings[3] += np.array([-1.0] * d_k)  # token 3 ve 5 benzer
    embeddings[5] += np.array([-1.0] * d_k)

    # Öğrenilmiş projeksiyon matrisleri (rastgele)
    W_Q = np.random.randn(d_k, d_k) * 0.1
    W_K = np.random.randn(d_k, d_k) * 0.1

    Q = embeddings @ W_Q
    K = embeddings @ W_K
    V = embeddings.copy()

    # Batch boyutu ekle
    Q_b = Q[np.newaxis]
    K_b = K[np.newaxis]
    V_b = V[np.newaxis]

    _, weights = scaled_dot_product_attention(Q_b, K_b, V_b)
    W = weights[0]

    print("Attention ağırlık matrisi (satır=query, sütun=key):")
    print(np.round(W, 3))
    print("\nEn yüksek attention (her token için):")
    for i in range(n):
        top_j = np.argmax(W[i])
        print(f"  Token {i} en çok Token {top_j}'e bakıyor (w={W[i, top_j]:.4f})")


# ─────────────────────────────────────────────────────────────
# 6. PYTORCH İMPLEMENTASYONU
# ─────────────────────────────────────────────────────────────

def pytorch_attention():
    print("\n" + "=" * 60)
    print("6. PYTORCH SCALED DOT-PRODUCT ATTENTION")
    print("=" * 60)

    try:
        import torch
        import torch.nn.functional as F

        torch.manual_seed(42)
        batch, seq, d_k, d_v = 2, 10, 64, 64

        Q = torch.randn(batch, seq, d_k)
        K = torch.randn(batch, seq, d_k)
        V = torch.randn(batch, seq, d_v)

        # PyTorch 2.0+: scaled_dot_product_attention (FlashAttention destekli)
        # causal mask
        output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        print(f"PyTorch SDPA: {Q.shape} → {output.shape}")

        # Elle implementasyon
        scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
        # Causal mask
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        output_manual = weights @ V
        print(f"Elle SDPA: {output_manual.shape}")
        print(f"PyTorch vs Manuel fark: {(output - output_manual).abs().max().item():.2e}")

        # Parametre sayısı analizi
        # Self-attention için: W_Q, W_K, W_V, W_O ∈ R^{d x d}
        d_model = 512
        attn_params = 4 * d_model * d_model  # Q, K, V, O projektionları
        print(f"\nSelf-attention (d={d_model}): {attn_params:,} parametre")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    motivasyon()
    qkv_sezgisi()
    attention_adim_adim()
    causal_mask_demo()
    attention_pattern_analizi()
    pytorch_attention()
