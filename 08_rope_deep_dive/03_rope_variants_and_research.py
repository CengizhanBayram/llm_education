"""
=============================================================
MODÜL 8.3 — RoPE VARYANTLARı VE ARAŞTIRMA KONULARI
=============================================================

Bu dosya LLM researcher için ileri düzey RoPE araştırma konularını
kapsar. Buradaki bilgiler doğrudan araştırma çalışmalarına yönlendirir.

Konular:
  1. xPos — RoPE'nin üstel bozunma ile genişletilmesi
  2. ALiBi vs RoPE derinlemesine karşılaştırma
  3. RoPE ve attention pattern analizi
  4. KV Cache ile RoPE etkileşimi
  5. RoPE'deki açık araştırma soruları
  6. Kendi RoPE varyantını kodlama şablonu
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


# ─────────────────────────────────────────────────────────────
# 1. xPos — eXtended Positional Encoding
# ─────────────────────────────────────────────────────────────
# Sun et al. 2022 "A Length-Extrapolatable Representation"
#
# Fikir: RoPE'ye üstel bozunma (exponential decay) ekle
#   Uzak tokenlardan gelen sinyal doğal olarak zayıflasın.
#
# Formül: xPos(q, m) = RoPE(q, m) ⊙ γ^m
#   γ: bozunma faktörü (element-wise, her boyut için farklı)
#   γ_i = (1 - 0.4/d) + (0.4/d) * i   (boyutlara göre 0.6 ~ 1.0)
#
# q · k = (γ^m q_rope) · (γ^(-n) k_rope) = γ^{m-n} q_rope · k_rope
#       = γ^{|m-n|} × (standart RoPE skoru)
#
# Uzak tokenlar için γ^{büyük} → skor düşüyor
# → Uzun sekanslarda stabil!

class xPos:
    """
    xPos: RoPE + üstel bozunma.
    Query için γ^m, Key için γ^{-n} çarpılır.
    """
    def __init__(self, d_k, base=10000.0):
        self.d_k = d_k
        i = np.arange(d_k // 2)
        self.theta = base ** (-2 * i / d_k)

        # Bozunma faktörü: her boyut için farklı
        # Düşük boyutlar (uzun mesafe): γ ≈ 1 (az bozunma)
        # Yüksek boyutlar (kısa mesafe): γ ≈ 0.6 (hızlı bozunma)
        self.gamma = (1 - 0.4/d_k) + (0.4/d_k) * i  # (d_k/2,) in [0.6, 1.0)

    def _rope(self, x, pos):
        """RoPE döndürmesi."""
        seq = x.shape[0]
        m_theta = np.outer(pos if hasattr(pos, '__len__') else [pos],
                           self.theta)   # (seq, d/2)
        cos_f = np.repeat(np.cos(m_theta), 2, axis=-1)
        sin_f = np.repeat(np.sin(m_theta), 2, axis=-1)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        rot = np.zeros_like(x)
        rot[..., 0::2] = -x_odd
        rot[..., 1::2] =  x_even
        return x * cos_f + rot * sin_f

    def apply_to_q(self, q, positions):
        """q için: RoPE × γ^m"""
        q_rope = self._rope(q, positions)
        # γ_i^m için: her pozisyon için ayrı bozunma
        decay = np.array([np.repeat(self.gamma ** positions[i], 2)
                          for i in range(len(positions))])  # (seq, d_k)
        return q_rope * decay

    def apply_to_k(self, k, positions):
        """k için: RoPE × γ^{-n}"""
        k_rope = self._rope(k, positions)
        decay = np.array([np.repeat(self.gamma ** (-positions[i]), 2)
                          for i in range(len(positions))])
        return k_rope * decay


def xpos_demo():
    print("=" * 65)
    print("1. xPos — UZATILMIŞ POZİSYONEL KODLAMA")
    print("=" * 65)

    np.random.seed(42)
    d_k = 16
    xpos = xPos(d_k)

    print(f"xPos bozunma faktörleri γ_i (d_k={d_k}):")
    for idx, g in enumerate(xpos.gamma):
        print(f"  i={idx}: γ = {g:.4f}")

    # Uzak tokenlar için bozunma etkisi
    q = np.random.randn(1, d_k)
    k = np.random.randn(1, d_k)
    positions_q = np.array([0])   # q her zaman konum 0

    print("\nFarklı mesafeler için xPos vs RoPE iç çarpımı:")
    print(f"{'Mesafe (n)':>12}  {'RoPE skoru':>14}  {'xPos skoru':>14}  {'Bozunma':>10}")
    print("-" * 54)

    # Temel RoPE referans
    from_rope = __import__('numpy')  # dummy import
    i = np.arange(d_k // 2)
    theta = 10000.0 ** (-2 * i / d_k)

    def rope_score(q, k, m, n, theta):
        def rope(v, pos):
            m_th = pos * theta
            cos_f = np.repeat(np.cos(m_th), 2)
            sin_f = np.repeat(np.sin(m_th), 2)
            v_e, v_o = v[0::2], v[1::2]
            r = np.zeros(d_k)
            r[0::2] = -v_o; r[1::2] = v_e
            return v * cos_f + r * sin_f
        return np.dot(rope(q[0], m), rope(k[0], n))

    for n in [0, 1, 5, 10, 50, 100, 500]:
        positions_k = np.array([n])
        q_xp = xpos.apply_to_q(q, positions_q)
        k_xp = xpos.apply_to_k(k, positions_k)
        score_xpos = np.dot(q_xp[0], k_xp[0])
        score_rope = rope_score(q, k, 0, n, theta)
        bozunma = np.prod(xpos.gamma ** n)  # ortalama etki
        print(f"{n:>12}  {score_rope:>14.6f}  {score_xpos:>14.6f}  {bozunma:>10.6f}")

    print("\n→ xPos: uzak tokenlar doğal olarak daha az katkıda bulunuyor")
    print("→ RoPE: mesafeye göre doğrudan decay yok")


# ─────────────────────────────────────────────────────────────
# 2. ALiBi vs RoPE DETAYLI KARŞILAŞTIRMA
# ─────────────────────────────────────────────────────────────

def alibi_vs_rope():
    print("\n" + "=" * 65)
    print("2. ALiBi vs RoPE DETAYLI KARŞILAŞTIRMA")
    print("=" * 65)

    print("""
  ALiBi (Press et al. 2021):
    score(i,j) = q_i · k_j / √d_k  -  m_h · |i - j|
    m_h: her kafa için slope (2^{-8h/n_heads})

  RoPE (Su et al. 2021):
    score(i,j) = RoPE(q_i, i) · RoPE(k_j, j) / √d_k
               = f(q_i, k_j, i-j)  (göreceli)

  ──────────────────────────────────────────────────────────
  ALiBi güçlü yönleri:
    ✓ Ekstra parametre yok (embedding tablosu bile yok)
    ✓ Attention hesabına sadece bias ekleniyor
    ✓ Uzun sekanslara basit extrapolation
    ✓ Eğitim sırasında görülmemiş uzunluklara sıfır maliyet

  ALiBi zayıf yönleri:
    ✗ Göreceli pozisyon bilgisi kaba (lineer penalty)
    ✗ Absolute position bilgisi yok
    ✗ Fine-tuning ile uzatmak zorunda değil ama kalitesi RoPE'den düşük
    ✗ Büyük modellerde (<7B) RoPE'ye yetişemiyor

  ──────────────────────────────────────────────────────────
  RoPE güçlü yönleri:
    ✓ Göreceli pozisyon doğal olarak kodlanıyor
    ✓ Rotasyon → norm koruyan → kararlı aktivasyon büyüklükleri
    ✓ Büyük modellerde üstün performans
    ✓ YaRN/NTK ile context uzatılabiliyor

  RoPE zayıf yönleri:
    ✗ Context uzatmak için fine-tune veya özel teknik gerekiyor
    ✗ Hesaplama: cos/sin → küçük ek yük
    ✗ KV cache'e uygulanması dikkat ister

  Neden RoPE kazandı? (2023-2024 tüm büyük modeller RoPE):
    1. LLaMA'nın popülerliği (açık ağırlık etkisi)
    2. YaRN ile context uzatma sorunu çözüldü
    3. Flash Attention ile iyi entegrasyon
    4. Büyük ölçeklerde (>7B) daha iyi benchmark sonuçları
    """)

    # Sayısal karşılaştırma: 8 kafa için ALiBi slopeları
    n_heads = 8
    slopes = [2 ** (-8 * (h+1) / n_heads) for h in range(n_heads)]
    print("ALiBi slopeları (8 kafa):")
    for h, s in enumerate(slopes):
        print(f"  Kafa {h}: m = {s:.4f}  (|i-j|=10'da ceza = {s*10:.4f})")


# ─────────────────────────────────────────────────────────────
# 3. RoPE VE ATTENTION PATTERN ANALİZİ
# ─────────────────────────────────────────────────────────────
# RoPE eğitilmiş modellerde attention nasıl şekilleniyor?

def rope_attention_pattern():
    print("\n" + "=" * 65)
    print("3. RoPE VE ATTENTION PATTERN ANALİZİ")
    print("=" * 65)

    np.random.seed(7)
    d_k, n_heads, seq = 32, 4, 12
    base = 10000.0

    i = np.arange(d_k // 2)
    theta = base ** (-2 * i / d_k)

    def apply_rope(x, seq_len, theta):
        """x: (seq, d_k)"""
        pos = np.arange(seq_len)
        angles = np.outer(pos, theta)   # (seq, d/2)
        cos_f = np.repeat(np.cos(angles), 2, axis=-1)
        sin_f = np.repeat(np.sin(angles), 2, axis=-1)
        x_e = x[..., 0::2]; x_o = x[..., 1::2]
        rot = np.zeros_like(x)
        rot[..., 0::2] = -x_o; rot[..., 1::2] = x_e
        return x * cos_f + rot * sin_f

    # Her kafanın attention entropisi (ne kadar "dağınık" dikkat)
    print("RoPE'nin kafa entropisi üzerindeki etkisi:")
    print(f"{'Kafa':>6}  {'RoPE ile H':>14}  {'RoPE olmadan H':>16}")
    print("-" * 40)

    def softmax(x): return np.exp(x - x.max()) / np.exp(x - x.max()).sum()
    def entropy(p): return -np.sum(p * np.log(p + 1e-9))

    for head in range(n_heads):
        Wq = np.random.randn(d_k, d_k) * 0.1
        Wk = np.random.randn(d_k, d_k) * 0.1
        X  = np.random.randn(seq, d_k)

        Q = X @ Wq; K = X @ Wk

        # RoPE olmadan
        scores_plain = (Q @ K.T) / np.sqrt(d_k)
        mask = np.tril(np.ones((seq, seq)))
        scores_plain = np.where(mask.astype(bool), scores_plain, -1e9)
        w_plain = np.array([softmax(scores_plain[i, :i+1]) for i in range(seq)])
        H_plain = np.mean([entropy(w_plain[i]) for i in range(seq)])

        # RoPE ile
        Q_r = apply_rope(Q, seq, theta)
        K_r = apply_rope(K, seq, theta)
        scores_rope = (Q_r @ K_r.T) / np.sqrt(d_k)
        scores_rope = np.where(mask.astype(bool), scores_rope, -1e9)
        w_rope = np.array([softmax(scores_rope[i, :i+1]) for i in range(seq)])
        H_rope = np.mean([entropy(w_rope[i]) for i in range(seq)])

        print(f"{head:>6}  {H_rope:>14.4f}  {H_plain:>16.4f}")

    print("\n→ RoPE: genellikle daha düşük entropi → daha keskin (focused) attention")
    print("→ Özellikle yakın tokenlar arası ilişki daha güçlü kodlanıyor")


# ─────────────────────────────────────────────────────────────
# 4. KV CACHE İLE RoPE ETKİLEŞİMİ
# ─────────────────────────────────────────────────────────────

def kv_cache_rope():
    print("\n" + "=" * 65)
    print("4. KV CACHE İLE RoPE ETKİLEŞİMİ")
    print("=" * 65)

    print("""
  KV Cache: Önceki K ve V değerlerini saklayarak her adımda
            yalnızca yeni token'ı hesapla.

  RoPE + KV Cache sorunu:
    K değerleri cache'lenirken RoPE uygulanmış mı olmalı?
    → EVET: K'yı cache'lerken RoPE'yi uygula.
    → Neden? RoPE K'ya pozisyon bilgisini enjekte eder,
             bu bilgi tahmin anında sabit.

  Doğru prosedür:
    1. Yeni token t için: q = RoPE(Wq·x_t, pos=t)
    2. Yeni K: k_t = RoPE(Wk·x_t, pos=t)
    3. Cache'e ekle: K_cache[t] = k_t
    4. Attention: q · K_cache[:t+1]^T

  Yanlış prosedür (kaçınılması gereken):
    Cache'lenmiş K'ya sonradan RoPE uygulamak → pozisyon kayması!

  Sliding Window Attention (Mistral):
    KV cache sabit tutulur, eski tokenlar atılır.
    RoPE pozisyonları sliding window ile uyumlu olmalı:
    → "Sink tokens" ilk tokenlara sabit yüksek attention verir.
    """)

    # KV cache boyutu hesabı
    print("KV Cache bellek analizi:")
    configs = [
        ("LLaMA-3 8B",  32, 8,  128, 8192),
        ("LLaMA-3 70B", 80, 8,  128, 8192),
        ("Mistral 7B",  32, 8,  128, 4096),   # sliding window
    ]
    print(f"{'Model':20s}  {'ctx':>8}  {'KV Cache (BF16)':>18}")
    print("-" * 50)
    for name, L, n_kv, d_k, ctx in configs:
        cache_bytes = L * n_kv * ctx * d_k * 2 * 2  # K+V, BF16=2 bytes
        print(f"{name:20s}  {ctx:>8,}  {cache_bytes/1e9:>16.2f} GB")


# ─────────────────────────────────────────────────────────────
# 5. AÇIK ARAŞTIRMA SORULARI
# ─────────────────────────────────────────────────────────────

def arastirma_sorulari():
    print("\n" + "=" * 65)
    print("5. AÇIK ARAŞTIRMA SORULARI")
    print("=" * 65)

    print("""
  LLM researcher için ilginç RoPE araştırma soruları:

  ── Temel Sorular ────────────────────────────────────────────
  1. Optimal base değeri nedir?
     LLaMA-3: 500K, bazı modeller 1M kullanıyor.
     → Daha büyük base her zaman daha iyi mi?
     → Hesap maliyeti etkisi var mı?

  2. d_k (head dimension) ile frekans aralığı nasıl etkileşiyor?
     d_k büyüdükçe frekans aralığı genişliyor (daha fazla boyut).
     → GQA ile d_k değişince RoPE performance değişiyor mu?

  ── Mimari Sorular ────────────────────────────────────────────
  3. Cross-layer RoPE paylaşımı mümkün mü?
     Tüm katmanlar aynı RoPE'yi mi kullanmalı?
     → Bazı çalışmalar katmana-özel base öneriyor.

  4. RoPE + Sparse Attention kombinasyonu?
     BigBird, Longformer gibi modellerde RoPE ile sliding window
     nasıl optimal birleştirilir?

  5. 2D/3D RoPE?
     Vision-Language modellerinde görüntü tokenları için
     2D pozisyon gerekiyor (satır, sütun).
     → 2D RoPE: iki ayrı frekans seti kullan.
     → LLaMA + Vision çalışmaları bu soruyu araştırıyor.

  ── Teorik Sorular ────────────────────────────────────────────
  6. Görecelilik özelliği gerçekten gerekli mi?
     Empirik olarak sınandı mı? RoPE olmadan sadece göreceli
     pozisyon bilgisi kullanan alternatifler?

  7. RoPE ile eğitilen modelin "konum" temsili yorumlanabilir mi?
     Mechanistic interpretability: hangi katmanlar hangi pozisyon
     bilgisini ne zaman kullanıyor?

  ── Pratik Sorular ────────────────────────────────────────────
  8. YaRN parametrelerini veriyle öğrenmek?
     β_fast, β_slow, s parametreleri şimdi elle ayarlanıyor.
     → Gradient ile optimize edilebilir mi?

  9. Uzun context'te token önemi ölçümü?
     4096 ile 131072 context arasındaki token
     attention dağılımı nasıl değişiyor?
    """)


# ─────────────────────────────────────────────────────────────
# 6. KENDİ RoPE VARYANTINı KODLAMA ŞABLONU
# ─────────────────────────────────────────────────────────────

def rope_varyant_sablonu():
    print("\n" + "=" * 65)
    print("6. KENDİ RoPE VARYANTINı KODLAMA ŞABLONU")
    print("=" * 65)

    if not TORCH:
        print("PyTorch gerekli.")
        return

    print("Yeni bir RoPE varyantı için şablon:")

    class CustomRoPE(nn.Module):
        """
        Kendi RoPE varyantını uygulamak için şablon.
        Değiştirilebilir noktalar:
          - _compute_theta(): frekans hesabı
          - _apply_rotation(): döndürme işlemi
          - forward(): tam pipeline
        """
        def __init__(self, d_k, max_seq=4096,
                     base=10000.0,
                     scale_factor=1.0,   # PI için: target_ctx/train_ctx
                     mode='standard'):   # 'standard', 'pi', 'ntk', 'yarn'
            super().__init__()
            self.d_k   = d_k
            self.scale = scale_factor
            self.mode  = mode

            theta = self._compute_theta(d_k, base, scale_factor, mode)
            self.register_buffer('theta', theta)

            positions = torch.arange(max_seq).float()
            angles = torch.outer(positions, theta)
            self.register_buffer('cos_cache', angles.cos())
            self.register_buffer('sin_cache', angles.sin())

        def _compute_theta(self, d_k, base, scale, mode):
            i = torch.arange(d_k // 2).float()

            if mode == 'standard':
                theta = 1.0 / (base ** (2 * i / d_k))

            elif mode == 'pi':
                # Position Interpolation: frekansı scale ile böl
                theta = 1.0 / (base ** (2 * i / d_k)) / scale

            elif mode == 'ntk':
                # NTK-aware: base'i artır
                new_base = base * (scale ** (d_k / (d_k - 2)))
                theta = 1.0 / (new_base ** (2 * i / d_k))

            elif mode == 'yarn':
                # YaRN: düşük/orta/yüksek frekans ayrımı
                beta_fast = 32; beta_slow = 1
                theta_orig = 1.0 / (base ** (2 * i / d_k))
                periods = 2 * np.pi / theta_orig.numpy()
                theta = theta_orig.clone()
                for idx in range(d_k // 2):
                    p = periods[idx]
                    if p < beta_fast:
                        theta[idx] = theta_orig[idx] / scale
                    elif p > beta_slow:
                        pass  # değişmez
                    else:
                        t = (p - beta_slow) / (beta_fast - beta_slow)
                        theta[idx] = theta_orig[idx] / (t * scale + (1-t))

            return theta

        @staticmethod
        def _rotate_half(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack([-x2, x1], dim=-1).flatten(-2)

        def forward(self, x, offset=0):
            """x: (batch, n_heads, seq, d_k)"""
            seq = x.shape[2]
            cos = self.cos_cache[offset:offset+seq].repeat_interleave(2, -1)
            sin = self.sin_cache[offset:offset+seq].repeat_interleave(2, -1)
            cos = cos[None, None]
            sin = sin[None, None]
            return x * cos + self._rotate_half(x) * sin

    # Karşılaştırma
    torch.manual_seed(0)
    d_k, seq = 64, 20

    modes = [
        ('standard', 1.0),
        ('pi',       4.0),
        ('ntk',      4.0),
        ('yarn',     4.0),
    ]

    print(f"\n{'Mod':12s}  {'scale':>8}  {'İlk θ':>12}  {'Son θ':>12}")
    print("-" * 50)
    for mode, scale in modes:
        rope = CustomRoPE(d_k, base=10000.0, scale_factor=scale, mode=mode)
        theta_arr = rope.theta.numpy()
        print(f"{mode:12s}  {scale:>8.1f}  {theta_arr[0]:>12.6f}  {theta_arr[-1]:>12.8f}")

    # Forward pass
    x = torch.randn(2, 4, seq, d_k)
    for mode, scale in modes:
        rope = CustomRoPE(d_k, base=10000.0, scale_factor=scale, mode=mode)
        out = rope(x)
        assert out.shape == x.shape
        assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5), "Norm korunmadı!"
    print("\nTüm modlarda norm korunuyor ✓")


if __name__ == "__main__":
    xpos_demo()
    alibi_vs_rope()
    rope_attention_pattern()
    kv_cache_rope()
    arastirma_sorulari()
    rope_varyant_sablonu()
