"""
=============================================================
MODÜL 8.2 — RoPE UZANTILARI (Context Length Scaling)
=============================================================

Temel RoPE sorun: Eğitim maksimum bağlamını aşınca performans düşer.
  LLaMA-2 7B eğitimi: 4096 token
  İnference: 4097+ token → aniden bozulma

Çözüm yöntemleri:
  1. Position Interpolation (PI) — basit lineer ölçekleme
  2. YaRN (Yet another RoPE extensioN) — LLaMA'nın resmi yöntemi
  3. LongRoPE — tek frekans başına farklı ölçekleme
  4. LLaMA-3 RoPE: base=500000 + NTK-aware

Her yöntem matematiksel türetim + NumPy implementasyonu içerir.
=============================================================
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# Temel RoPE yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────

def build_rope_cache(max_seq, d_k, base=10000.0):
    """cos/sin önbelleği: (max_seq, d_k/2)"""
    i = np.arange(d_k // 2)
    theta = base ** (-2 * i / d_k)
    pos   = np.arange(max_seq)
    angles = np.outer(pos, theta)   # (max_seq, d_k/2)
    return np.cos(angles), np.sin(angles)

def apply_rope_numpy(x, cos_cache, sin_cache, offset=0):
    """
    x: (batch, n_heads, seq, d_k)
    """
    seq = x.shape[2]
    cos = cos_cache[offset:offset+seq]   # (seq, d/2)
    sin = sin_cache[offset:offset+seq]

    cos_full = np.repeat(cos, 2, axis=-1)[np.newaxis, np.newaxis]  # (1,1,seq,d)
    sin_full = np.repeat(sin, 2, axis=-1)[np.newaxis, np.newaxis]

    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    rot = np.zeros_like(x)
    rot[..., 0::2] = -x_odd
    rot[..., 1::2] =  x_even
    return x * cos_full + rot * sin_full

def dot_product_score(q_rot, k_rot, pos_m, pos_n):
    """Konum m'deki q ile konum n'deki k'nın iç çarpımı."""
    return float(np.dot(q_rot[0, 0, pos_m], k_rot[0, 0, pos_n]))


# ─────────────────────────────────────────────────────────────
# 1. TEMEL SORUN: CONTEXT DIŞI DEGRADASYON
# ─────────────────────────────────────────────────────────────

def context_disindaki_sorun():
    print("=" * 65)
    print("1. TEMEL SORUN: CONTEXT DIŞI DEGRADASYON")
    print("=" * 65)

    print("""
  LLaMA-2 7B: eğitim context = 4096 token
  Eğitim sırasında model hiç 4097+ konum görmedi.

  Sorun: RoPE pozisyon 4097+ için cos/sin değerleri hesaplanabilir
          AMA model bu değerlerle başa çıkmayı öğrenmedi!
          → Attention skorları tutarsız → coherence bozulur

  Sezgi:
    RoPE frekansları θ_i için eğitim sırasında m·θ_i ≤ 4096·θ_i
    Düşük i (düşük frekans): m·θ_i küçük kalır → sorun az
    Yüksek i (yüksek frekans): θ_i büyük → m·θ_i büyür → döngü tamamlanır

  Örnek (d=128, base=10000):
    θ_0 = 1.0 → m=4096 → 4096·1.0 = 4096 radyan (birçok tur!)
    θ_63 ≈ 1e-4 → m=4096 → 0.4 radyan (çeyrek tur bile değil)
    """)

    # Ne kadar tur tamamlandı?
    d_k, base = 128, 10000.0
    train_len = 4096
    i = np.arange(d_k // 2)
    theta = base ** (-2 * i / d_k)

    print("Eğitim sonu (m=4096) rotasyon miktarı:")
    print(f"{'Boyut i':>8}  {'θ_i':>12}  {'4096·θ_i (rad)':>16}  {'Tur sayısı':>12}")
    print("-" * 54)
    for idx in [0, 8, 16, 32, 48, 62, 63]:
        rad = train_len * theta[idx]
        turns = rad / (2 * np.pi)
        print(f"{idx:>8}  {theta[idx]:>12.6f}  {rad:>16.2f}  {turns:>12.2f}")

    print("\n→ Yüksek boyutlar (büyük i) çok az tur → eğitimde az bilgi")
    print("→ Uzatma için bu boyutlar daha 'uysal'")


# ─────────────────────────────────────────────────────────────
# 2. POSITION INTERPOLATION (PI)
# ─────────────────────────────────────────────────────────────
# Chen et al. 2023 "Extending Context Window of LLMs via PI"
#
# Fikir: Konum indeksini ölçekle
#   Orijinal RoPE:  f(x, m) = x · e^{i·m·θ}
#   PI:             f(x, m) = x · e^{i·(m/s)·θ}
#
#   s: ölçekleme faktörü = hedef_ctx / eğitim_ctx
#   Örn: 4096 → 16384 için s = 4
#
# Sezgi:
#   m = 16384 token konumu → m/s = 4096 → eğitimde görülen aralık
#   Model eğitimde gördüğü θ değerlerini kullanıyor → daha kararlı
#
# Sorun:
#   Kısa mesafelerde de sıkıştırma var → yakın tokenları ayırt etmek güçleşir
#   Biraz fine-tune gerekiyor (1000 adım yeterli)

def position_interpolation():
    print("\n" + "=" * 65)
    print("2. POSITION INTERPOLATION (PI)")
    print("=" * 65)

    print("""
  Orijinal: f(x, m, θ) = x · e^{i·m·θ}
  PI:       f(x, m, θ) = x · e^{i·(m/s)·θ}

  s = target_ctx / train_ctx

  Eşdeğer yorum: θ_eff = θ / s (frekansı küçült)
  """)

    d_k = 64
    base = 10000.0
    train_ctx = 4096
    target_ctx = 16384
    s = target_ctx / train_ctx   # = 4

    # Orijinal θ
    i = np.arange(d_k // 2)
    theta_orig = base ** (-2 * i / d_k)

    # PI ile θ (frekans küçültme)
    theta_pi = theta_orig / s

    print(f"Ölçekleme faktörü: s = {target_ctx}/{train_ctx} = {s}")
    print(f"\nFrekans karşılaştırması:")
    print(f"{'Boyut i':>8}  {'θ_orig':>12}  {'θ_PI':>12}  {'Oran':>8}")
    print("-" * 46)
    for idx in [0, 16, 31]:
        print(f"{idx:>8}  {theta_orig[idx]:>12.6f}  {theta_pi[idx]:>12.6f}  {theta_pi[idx]/theta_orig[idx]:>8.3f}")

    # m=16384 konumunda iki yöntemin karşılaştırması
    m_new = 16384
    print(f"\nm={m_new} konumunda açılar:")
    print(f"{'Boyut i':>8}  {'Orijinal m·θ':>16}  {'PI (m/s)·θ':>16}")
    print("-" * 44)
    for idx in [0, 31]:
        angle_orig = m_new * theta_orig[idx]
        angle_pi   = (m_new / s) * theta_orig[idx]
        print(f"{idx:>8}  {angle_orig:>16.2f}  {angle_pi:>16.2f}  "
              f"{'(eğitim aralığında ✓)' if angle_pi <= train_ctx * theta_orig[idx] else ''}")


# ─────────────────────────────────────────────────────────────
# 3. NTK-AWARE SCALING
# ─────────────────────────────────────────────────────────────
# bloc97/NTK-aware (2023) — PI'nin geliştirilmiş versiyonu
#
# Fikir: Base'i artır, tüm frekansları eşit sıkıştırma yerine
#        yüksek frekansları daha az dokunarak uzat.
#
# NTK (Neural Tangent Kernel) teorisinden esinlenildi:
#   base_new = base * s^(d/(d-2))
#
# Örn: 4096→16384, d=128:
#   base_new = 10000 * 4^(128/126) ≈ 41068
#
# Avantaj: Ağırlık değişikliği yok (fine-tune gerekmez!)
# Dezavantaj: PI kadar iyi değil, ama sıfır maliyet

def ntk_aware_scaling():
    print("\n" + "=" * 65)
    print("3. NTK-AWARE SCALING")
    print("=" * 65)

    d_k = 128
    base = 10000.0
    train_ctx = 4096
    target_ctx = 32768
    s = target_ctx / train_ctx

    # NTK base formülü
    base_ntk = base * (s ** (d_k / (d_k - 2)))
    print(f"Train ctx: {train_ctx}  →  Target ctx: {target_ctx}")
    print(f"s = {s:.1f}")
    print(f"NTK base = {base} × {s:.1f}^({d_k}/{d_k-2}) = {base_ntk:.0f}")

    i = np.arange(d_k // 2)
    theta_orig = base     ** (-2 * i / d_k)
    theta_ntk  = base_ntk ** (-2 * i / d_k)
    theta_pi   = theta_orig / s

    print(f"\nFrekans karşılaştırması (d_k={d_k}):")
    print(f"{'Boyut i':>8}  {'Orijinal':>12}  {'PI':>12}  {'NTK':>12}")
    print("-" * 50)
    for idx in [0, 16, 32, 48, 63]:
        print(f"{idx:>8}  {theta_orig[idx]:>12.6f}  {theta_pi[idx]:>12.6f}  {theta_ntk[idx]:>12.6f}")

    print(f"\nNTK: düşük i'de PI gibi, yüksek i'de orijinale yakın")
    print("→ Yüksek frekanslı boyutlar az sıkışıyor → kısa mesafe bilgisi korunuyor")


# ─────────────────────────────────────────────────────────────
# 4. YaRN — Yet Another RoPE extensioN
# ─────────────────────────────────────────────────────────────
# Peng et al. 2023 — LLaMA için resmi yöntem
#
# Fikir: Her frekans grubuna farklı muamele:
#   - Düşük frekanslar (geniş periyot): dokunma (zaten uzun mesafeyi kapsıyor)
#   - Yüksek frekanslar (dar periyot):  NTK ile ölçekle
#   - Orta frekanslar: doğrusal interpolasyon
#
# Matematiği:
#   β: frekansı düşük/orta/yüksek'e ayıran eşik
#
#   low_freq:  θ_i → θ_i (değişmez)
#   high_freq: θ_i → θ_i / s (PI ile aynı)
#   mid_freq:  θ_i → θ_i / (1 - (1/s - 1) * (β / θ_i - 1))  (yumuşak geçiş)
#
# Ayrıca "attention temperature" düzeltmesi:
#   Uzatılmış model: softmax(QK^T / (√d_k · t))  burada t = 0.1*ln(s) + 1

def yarn_extension():
    print("\n" + "=" * 65)
    print("4. YaRN — YET ANOTHER RoPE EXTENSION")
    print("=" * 65)

    d_k = 128
    base = 10000.0
    train_ctx = 4096
    target_ctx = 32768
    s = target_ctx / train_ctx

    # YaRN parametreleri (LLaMA orijinal değerleri)
    beta_fast = 32    # yüksek frekans eşiği (periyot < bu → high freq)
    beta_slow = 1     # düşük frekans eşiği (periyot > bu → low freq)

    i = np.arange(d_k // 2)
    theta_orig = base ** (-2 * i / d_k)
    periyots = 2 * np.pi / theta_orig  # token cinsinden periyot

    def yarn_scale(theta_i, periyot_i, s, beta_fast, beta_slow):
        """Tek bir frekans için YaRN ölçekleme faktörü döndür."""
        if periyot_i < beta_fast:
            # Yüksek frekans: PI gibi sıkıştır
            return theta_i / s
        elif periyot_i > beta_slow:
            # Düşük frekans: değişme
            return theta_i
        else:
            # Orta frekans: lineer interpolasyon
            low  = beta_slow
            high = beta_fast
            t = (periyot_i - low) / (high - low)
            return theta_i / (t * s + (1 - t))

    theta_yarn = np.array([yarn_scale(theta_orig[idx], periyots[idx], s, beta_fast, beta_slow)
                            for idx in range(len(theta_orig))])

    # Attention temperature
    t_yarn = 0.1 * np.log(s) + 1.0
    print(f"YaRN parametreleri:")
    print(f"  s={s:.1f},  beta_fast={beta_fast},  beta_slow={beta_slow}")
    print(f"  Attention temperature: t = 0.1·ln({s:.0f})+1 = {t_yarn:.4f}")
    print(f"  (√d_k → √d_k · t ile kullanılır)")

    # Frekans grupları
    n_low  = np.sum(periyots > beta_slow)
    n_mid  = np.sum((periyots >= beta_fast) & (periyots <= beta_slow))
    n_high = np.sum(periyots < beta_fast)
    print(f"\nFrekans grup dağılımı (d_k={d_k}):")
    print(f"  Düşük  (periyot > {beta_slow}):      {n_low} boyut  → değişmez")
    print(f"  Orta   ({beta_fast} ≥ periyot ≥ {beta_slow}): {n_mid} boyut  → interpolasyon")
    print(f"  Yüksek (periyot < {beta_fast}):    {n_high} boyut  → PI ile sıkıştır")

    print(f"\nFrekans karşılaştırması:")
    print(f"{'Boyut i':>8}  {'Periyot':>10}  {'θ_orig':>12}  {'θ_PI':>12}  {'θ_YaRN':>12}  {'Grup':>8}")
    print("-" * 68)
    for idx in [0, 8, 16, 32, 48, 55, 60, 63]:
        p = periyots[idx]
        grp = "LOW" if p > beta_slow else ("HIGH" if p < beta_fast else "MID")
        print(f"{idx:>8}  {p:>10.1f}  {theta_orig[idx]:>12.6f}  "
              f"{theta_orig[idx]/s:>12.6f}  {theta_yarn[idx]:>12.6f}  {grp:>8}")


# ─────────────────────────────────────────────────────────────
# 5. LLAMA-3 RoPE AYARLARI
# ─────────────────────────────────────────────────────────────
# LLaMA-3 (2024): base = 500,000 (orijinal 10,000'den 50x büyük)
#
# Neden?
#   Daha büyük base → tüm frekanslar daha yavaş → doğal uzun context
#   Fine-tune sırasında 8K→128K için YaRN benzeri uzatma kullanıldı.
#
# LLaMA-3.1 (128K context) için kullanılan parametreler:
#   base = 500000
#   factor = 8  (32K → 128K için 4x ek uzatma)
#   low_freq_factor = 1
#   high_freq_factor = 4
#   original_max_position = 8192

def llama3_rope():
    print("\n" + "=" * 65)
    print("5. LLaMA-3 RoPE AYARLARI")
    print("=" * 65)

    d_k = 128
    configs = {
        "LLaMA-2 7B (train)":  {"base": 10000,  "ctx": 4096},
        "LLaMA-3 8B (train)":  {"base": 500000, "ctx": 8192},
        "LLaMA-3.1 8B (128K)": {"base": 500000, "ctx": 131072},
    }

    i = np.arange(d_k // 2)
    print(f"{'Model':28s}  {'Maks periyot (en uzun boyut)':>30}")
    print("-" * 62)
    for name, cfg in configs.items():
        theta = cfg["base"] ** (-2 * i / d_k)
        periyots = 2 * np.pi / theta
        max_periyot = periyots[-1]
        ctx = cfg["ctx"]
        print(f"{name:28s}  {max_periyot:>30,.0f} token  "
              f"({'ctx/ctx_yeterli ✓' if max_periyot > ctx else 'ctx/yetersiz ✗'})")

    print(f"\nLLaMA-3 base=500000 seçiminin nedeni:")
    base_orig = 10000.0
    base_new  = 500000.0
    theta_old = base_orig ** (-2 * i / d_k)
    theta_new = base_new  ** (-2 * i / d_k)
    periods_old = 2 * np.pi / theta_old
    periods_new = 2 * np.pi / theta_new
    print(f"  LLaMA-2: en uzun periyot = {periods_old[-1]:,.0f} token")
    print(f"  LLaMA-3: en uzun periyot = {periods_new[-1]:,.0f} token")
    print(f"  → 8192 ctx için LLaMA-3 çok daha iyi kapsama")

    # LLaMA-3.1 için tam YaRN benzeri uzatma
    print(f"\nLLaMA-3.1 128K uzatma stratejisi:")
    print("""
  1. base = 500,000 (zaten geniş)
  2. YaRN benzeri uzatma:
       original_max = 8,192
       target_max = 131,072   (8,192 × 16)
  3. Uzatma factor = 131072 / 8192 = 16
  4. ~100B token fine-tune (8K → 128K)
    """)


# ─────────────────────────────────────────────────────────────
# 6. UZATMA YÖNTEMLERİ KARŞILAŞTIRMASI
# ─────────────────────────────────────────────────────────────

def karsilastirma():
    print("\n" + "=" * 65)
    print("6. UZATMA YÖNTEMLERİ KARŞILAŞTIRMASI")
    print("=" * 65)

    print("""
  ┌────────────────┬─────────────────┬────────────┬───────────────────────────┐
  │ Yöntem         │ Fine-tune       │ Kalite     │ Notlar                    │
  ├────────────────┼─────────────────┼────────────┼───────────────────────────┤
  │ Orijinal RoPE  │ Yok             │ Eğitim ctx │ Ötesinde bozulur          │
  ├────────────────┼─────────────────┼────────────┼───────────────────────────┤
  │ PI             │ ~1K adım        │ İyi        │ Yakın mesafe zorlaşır     │
  ├────────────────┼─────────────────┼────────────┼───────────────────────────┤
  │ NTK-aware      │ Gereksiz        │ Orta       │ Sıfır ek maliyet          │
  ├────────────────┼─────────────────┼────────────┼───────────────────────────┤
  │ NTK-by-parts   │ Gereksiz        │ İyi        │ NTK'nın geliştirilmişi    │
  ├────────────────┼─────────────────┼────────────┼───────────────────────────┤
  │ YaRN           │ ~400 adım       │ Çok iyi    │ LLaMA resmi yöntemi       │
  ├────────────────┼─────────────────┼────────────┼───────────────────────────┤
  │ LongRoPE      │ Var             │ Çok iyi    │ Her frekansa farklı scale │
  ├────────────────┼─────────────────┼────────────┼───────────────────────────┤
  │ Büyük base     │ Var (initial)   │ En iyi     │ LLaMA-3 yaklaşımı         │
  └────────────────┴─────────────────┴────────────┴───────────────────────────┘

  Pratik tavsiye (2024):
    - Sıfırdan eğitim: base=500000+ kullan
    - Mevcut modeli uzat: YaRN veya LongRoPE
    - Hızlı deneme: NTK-aware (fine-tune yok)
    """)


if __name__ == "__main__":
    context_disindaki_sorun()
    position_interpolation()
    ntk_aware_scaling()
    yarn_extension()
    llama3_rope()
    karsilastirma()
