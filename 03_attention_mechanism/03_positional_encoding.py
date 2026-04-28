"""
=============================================================
MODÜL 3.3 — POZİSYONEL KODLAMA (Positional Encoding)
=============================================================

Attention mekanizması konumdan bağımsızdır (permutation equivariant):
  - "cat sat" ve "sat cat" için aynı attention skorunu üretir!
  - Çözüm: token embedding'e konum bilgisi ekle

LLM'lerde kullanılan pozisyonel kodlamalar:
  - Orijinal Transformer (2017):  Sinüzoidal PE
  - BERT, GPT-2:                  Öğrenilmiş PE (Learned PE)
  - GPT-NeoX, LLaMA, Mistral:     RoPE (Rotary Position Embedding)
  - ALiBi:                         Attention bias'ı ile konum

Konular:
  1. Neden pozisyonel kodlama lazım?
  2. Sinüzoidal Positional Encoding
  3. Öğrenilmiş Pozisyon Embedding
  4. RoPE (Rotary Position Embedding)
  5. ALiBi
  6. Karşılaştırma
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. NEDEN GEREKLİ?
# ─────────────────────────────────────────────────────────────

def neden_gerekli():
    print("=" * 60)
    print("1. NEDEN POZİSYONEL KODLAMA GEREKLİ?")
    print("=" * 60)

    # Attention, pozisyondan bağımsız çalışır
    # softmax(QK^T) → hangi konumda olduğuna bakmaz
    # "The cat sat" ve "sat The cat" aynı attention'ı üretir

    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        return np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)

    np.random.seed(42)
    d_k = 4
    n = 3

    # Aynı embedding, farklı sıra
    e1 = np.array([1.0, 0.0, 0.0, 0.0])  # "The"
    e2 = np.array([0.0, 1.0, 0.0, 0.0])  # "cat"
    e3 = np.array([0.0, 0.0, 1.0, 0.0])  # "sat"

    seq_normal   = np.array([e1, e2, e3])  # [The, cat, sat]
    seq_shuffled = np.array([e3, e1, e2])  # [sat, The, cat]

    W_Q = np.random.randn(d_k, d_k) * 0.5
    W_K = np.random.randn(d_k, d_k) * 0.5

    Q_n = seq_normal   @ W_Q
    K_n = seq_normal   @ W_K
    Q_s = seq_shuffled @ W_Q
    K_s = seq_shuffled @ W_K

    scores_n = softmax(Q_n @ K_n.T / np.sqrt(d_k))
    scores_s = softmax(Q_s @ K_s.T / np.sqrt(d_k))

    print("Attention (normal sıra [The, cat, sat]):")
    print(np.round(scores_n, 4))
    print("\nAttention (karışık sıra [sat, The, cat]):")
    print(np.round(scores_s, 4))
    print("\n→ Sıra farklı ama embedding aynı → farklı attention deseni")
    print("→ Sıra bilgisini açıkça eklemeden model sırayı bilemez!")


# ─────────────────────────────────────────────────────────────
# 2. SİNÜZOİDAL POZİSYONEL KODLAMA
# ─────────────────────────────────────────────────────────────
# "Attention Is All You Need" (2017):
#
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# pos: token konumu (0, 1, 2, ...)
# i: boyut indeksi (0, 1, ..., d_model/2 - 1)
#
# Özellikleri:
#   - Öğrenilmiyor, sabit matematik formülü
#   - Kosinüs benzerliği: PE(pos+k) her zaman aynı lineer dönüşümle PE(pos)'dan
#   - Görecelilik (relative): sin(a+b) = sin(a)cos(b) + cos(a)sin(b) formülü ile
#     model göreceli pozisyonu öğrenebilir
#   - Uzun sekanslara genelleşir (eğitimde görülmemiş uzunluklara)
#   - Dezavantaj: mutlak pozisyon bilgisi verir, göreceli zayıf

def sinusoidal_pe(max_len, d_model):
    """
    Returns: PE ∈ R^{max_len x d_model}
    """
    PE = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, np.newaxis]        # (max_len, 1)
    i   = np.arange(0, d_model, 2)[np.newaxis, :]  # (1, d_model/2)

    # div_term: 10000^(2i/d_model)
    div_term = np.power(10000.0, i / d_model)

    PE[:, 0::2] = np.sin(pos / div_term)   # çift boyutlar: sin
    PE[:, 1::2] = np.cos(pos / div_term)   # tek boyutlar: cos
    return PE


def sinusoidal_demo():
    print("\n" + "=" * 60)
    print("2. SİNÜZOİDAL POZİSYONEL KODLAMA")
    print("=" * 60)

    d_model = 16
    max_len = 10

    PE = sinusoidal_pe(max_len, d_model)
    print(f"PE shape: {PE.shape}  (max_len={max_len}, d_model={d_model})")
    print(f"\nİlk 4 konum, ilk 8 boyut:")
    print(np.round(PE[:4, :8], 4))

    # Kosinüs benzerliği — konum 2 ve 5 arasındaki mesafe
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("\nKonum çiftleri arası kosinüs benzerliği:")
    pairs = [(0,1), (0,5), (0,9), (1,2), (4,5)]
    for i, j in pairs:
        sim = cosine_sim(PE[i], PE[j])
        print(f"  sim(pos={i}, pos={j}) = {sim:.4f}  (uzaklık={abs(i-j)})")
    print("→ Yakın konumlar birbirine daha benzer!")

    # Görecelilik: PE(pos+k) PE(pos) büyük
    print("\nGörecelilik doğrulaması:")
    pos, k = 3, 2
    # sin(a+b) = sin(a)cos(b) + cos(a)sin(b) özelliği ile
    # PE(pos+k)'yı PE(pos)'tan lineer dönüşümle elde edebiliriz
    print(f"  PE[{pos}] = {np.round(PE[pos, :4], 4)}")
    print(f"  PE[{pos+k}] = {np.round(PE[pos+k, :4], 4)}")
    print(f"  sim = {cosine_sim(PE[pos], PE[pos+k]):.4f}")


# ─────────────────────────────────────────────────────────────
# 3. ÖĞRENİLMİŞ POZİSYON EMBEDDING
# ─────────────────────────────────────────────────────────────
# GPT-2, BERT kullanır:
#   P ∈ R^{max_len x d_model}  — eğitimde öğrenilir
#
# Kullanım: x_t = token_embedding(w_t) + position_embedding(t)
#
# Avantaj: Eğitim verisinden en iyi pozisyon temsilini öğrenir
# Dezavantaj: max_len'i aşan sekanslara genelleşemez

def learned_pe_demo():
    print("\n" + "=" * 60)
    print("3. ÖĞRENİLMİŞ POZİSYON EMBEDDING")
    print("=" * 60)

    max_len = 1024    # GPT-2 max context length
    d_model = 768     # GPT-2 small
    vocab   = 50257

    token_emb = np.random.randn(vocab, d_model) * 0.02
    pos_emb   = np.random.randn(max_len, d_model) * 0.02

    print(f"Token embedding: {token_emb.shape}  ({token_emb.size:,} parametre)")
    print(f"Position embedding: {pos_emb.shape}  ({pos_emb.size:,} parametre)")

    # Örnek sekans: "Hello world" → token ids [15496, 995]
    token_ids = [15496, 995]
    seq_len = len(token_ids)

    x = np.array([token_emb[t] + pos_emb[p]
                  for p, t in enumerate(token_ids)])
    print(f"\nSekans '{token_ids}' embedding: {x.shape}")
    print(f"Toplam giriş = token_emb + pos_emb")
    print(f"GPT-2 small toplam embedding parametresi: {(vocab+max_len)*d_model:,}")


# ─────────────────────────────────────────────────────────────
# 4. RoPE — ROTARY POSITION EMBEDDING
# ─────────────────────────────────────────────────────────────
# Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
# Kullanım: LLaMA, LLaMA-2/3, Mistral, Falcon, GPT-NeoX ...
#
# Fikir: Mutlak konum bilgisini Q ve K'ya döndürme matrisiyle kodla.
# Sonuç: Q_m · K_n sadece (m-n) göreceli konuma bağlı olur!
#
# Formül:
# Bir çift (x_{2i}, x_{2i+1}) için konum m'de:
#   x'_{2i}   = x_{2i}   cos(m θ_i) - x_{2i+1} sin(m θ_i)
#   x'_{2i+1} = x_{2i}   sin(m θ_i) + x_{2i+1} cos(m θ_i)
#
# θ_i = 10000^(-2i/d)
#
# Bu, 2D düzlemde m*θ_i açısıyla döndürme işlemi.
#
# Avantajlar:
#   - Göreceli pozisyon doğal olarak kodlanır
#   - Uzun sekanslara (training dışı) iyi genelleşir (YaRN, LongRoPE ile daha iyi)
#   - Küçük hesap maliyeti
#   - Mutlak + göreceli bilginin en iyi kombinasyonu

def rope(x, seq_len, base=10000.0):
    """
    x: (seq_len, d_model)
    Döndürme işlemini uygula, döndürülmüş x döndür.
    """
    d = x.shape[-1]
    assert d % 2 == 0

    # θ_i = base^(-2i/d)
    i = np.arange(d // 2)
    theta = 1.0 / (base ** (2 * i / d))   # (d/2,)

    # m * θ_i için her konum
    positions = np.arange(seq_len)         # (seq,)
    angles = np.outer(positions, theta)    # (seq, d/2)

    cos_vals = np.cos(angles)   # (seq, d/2)
    sin_vals = np.sin(angles)

    # x'i çift/tek boyutlara böl
    x_even = x[:, 0::2]   # (seq, d/2)
    x_odd  = x[:, 1::2]   # (seq, d/2)

    # Döndürme:
    # [x_even * cos - x_odd * sin,
    #  x_even * sin + x_odd * cos]
    x_rotated_even = x_even * cos_vals - x_odd * sin_vals
    x_rotated_odd  = x_even * sin_vals + x_odd * cos_vals

    # Yeniden birleştir
    out = np.zeros_like(x)
    out[:, 0::2] = x_rotated_even
    out[:, 1::2] = x_rotated_odd
    return out


def rope_demo():
    print("\n" + "=" * 60)
    print("4. RoPE — ROTARY POSITION EMBEDDING")
    print("=" * 60)

    np.random.seed(42)
    seq_len, d_k = 6, 8

    # Q ve K oluştur
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)

    # RoPE uygula
    Q_rope = rope(Q, seq_len)
    K_rope = rope(K, seq_len)

    # RoPE'nun temel özelliği:
    # Q_m · K_n = f(m-n)  — sadece göreceli konuma bağlı

    # Doğrulama: Aynı göreceli mesafe → benzer iç çarpım
    def dot(a, b):
        return np.dot(a, b)

    print("RoPE ile Q_m · K_n örnekleri:")
    print("  (m, n) → dot product | göreceli mesafe")
    for m, n in [(0,0), (1,1), (2,2), (0,1), (1,2), (2,3), (0,3)]:
        d = dot(Q_rope[m], K_rope[n])
        print(f"  ({m},{n}): {d:>8.4f}  |  mesafe={abs(m-n)}")

    # Normal attention (konum bilgisi yok): aynı göreceli mesafe farklı değer
    print("\nNormal attention (PE yok), aynı göreceli mesafe:")
    for m, n in [(0,1), (1,2), (2,3)]:
        d = dot(Q[m], K[n])
        print(f"  ({m},{n}): {d:>8.4f}  |  mesafe=1 ama değer farklı")

    # RoPE frekans analizi
    print("\n--- RoPE FREKANS ANALİZİ ---")
    base = 10000.0
    d = 128  # LLaMA-2 d_k
    i = np.arange(d // 2)
    theta = 1.0 / (base ** (2*i/d))
    periods = 2 * np.pi / theta

    print(f"d_k = {d}")
    print(f"Boyut 0 (i=0): θ={theta[0]:.6f}, periyot={periods[0]:.2f} token")
    print(f"Boyut 32 (i={d//4}): θ={theta[d//4]:.6f}, periyot={periods[d//4]:.2f} token")
    print(f"Boyut 62 (i=62): θ={theta[62]:.6f}, periyot={periods[62]:.0f} token")
    print(f"→ Düşük boyutlar: kısa mesafe, Yüksek boyutlar: uzun mesafe")
    print(f"→ Bu hiyerarşi modele hem lokal hem global konum öğretir")


# ─────────────────────────────────────────────────────────────
# 5. ALiBi — Attention with Linear Biases
# ─────────────────────────────────────────────────────────────
# Press et al. (2021) — MPT, BLOOM kullanır
#
# Fikir: Attention skoruna uzaklık cezası ekle:
#   score(i,j) = q_i · k_j / √d_k - m * |i-j|
#
# m: kafa başına farklı slope (eğim)
#   m_h = 2^(-8 * h / n_heads)  →  h=1: 0.5, h=2: 0.25, ...
#
# Avantaj: Ekstra parametre yok, eğitimden uzun sekanslara genelleşir
# Dezavantaj: Absolute PE ile karşılaştırıldığında bazı görevlerde zayıf

def alibi_demo():
    print("\n" + "=" * 60)
    print("5. ALiBi — ATTENTION WITH LINEAR BIASES")
    print("=" * 60)

    n_heads = 8
    seq_len = 6

    # Her kafa için slope
    slopes = [2 ** (-8 * h / n_heads) for h in range(1, n_heads+1)]
    print(f"Kafa slopes: {[round(s, 4) for s in slopes]}")

    # Mesafe matrisi
    positions = np.arange(seq_len)
    dist_matrix = np.abs(positions[:, None] - positions[None, :])  # (seq, seq)
    print(f"\nMesafe matrisi:\n{dist_matrix}")

    # Kafa 0 için ALiBi bias
    bias_h0 = -slopes[0] * dist_matrix
    print(f"\nKafa 0 ALiBi bias (slope={slopes[0]:.4f}):\n{np.round(bias_h0, 3)}")

    # Kafa 7 için (daha yavaş decay)
    bias_h7 = -slopes[7] * dist_matrix
    print(f"\nKafa 7 ALiBi bias (slope={slopes[7]:.4f}):\n{np.round(bias_h7, 3)}")
    print("→ Küçük slope: uzak tokenlara hafif ceza (geniş bağlam)")
    print("→ Büyük slope: uzak tokenlara sert ceza (lokal dikkat)")


# ─────────────────────────────────────────────────────────────
# 6. POZİSYONEL KODLAMA KARŞILAŞTIRMASI
# ─────────────────────────────────────────────────────────────

def karsilastirma():
    print("\n" + "=" * 60)
    print("6. POZİSYONEL KODLAMA KARŞILAŞTIRMASI")
    print("=" * 60)

    print("""
  Yöntem         Modeller              Göreceli  Uzun sekans  Parametre
  ─────────────────────────────────────────────────────────────────────
  Sinüzoidal     Orijinal Transformer  Zayıf     İyi          0
  Öğrenilmiş    BERT, GPT-2           Yok       Kötü         max_len × d
  RoPE           LLaMA, Mistral        Güçlü     İyi*         0
  ALiBi          BLOOM, MPT            Orta      Çok İyi      0

  * YaRN veya LongRoPE ile uzatılabilir (örn. 8K → 128K context)

  Güncel tercih (2024):
    → RoPE: LLaMA-2, LLaMA-3, Mistral, Gemma, Qwen, Phi
    → RoPE + YaRN: uzun context (128K+)
    """)


if __name__ == "__main__":
    neden_gerekli()
    sinusoidal_demo()
    learned_pe_demo()
    rope_demo()
    alibi_demo()
    karsilastirma()
