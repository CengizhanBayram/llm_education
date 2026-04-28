"""
=============================================================
MODÜL 1.2 — OLASILIK & İSTATİSTİK
=============================================================

LLM'lerde neden olasılık teorisi?
  - Dil modeli → P(w_t | w_1, ..., w_{t-1}) olasılık dağılımını öğrenir
  - Softmax → logit'leri olasılık dağılımına dönüştürür
  - Sampling (temperature, top-k, top-p) → olasılık dağılımından örnekleme
  - MLE (Maximum Likelihood Estimation) → modelin eğitim hedefi

Konular:
  1. Temel olasılık kavramları
  2. Koşullu olasılık ve Bayes teoremi
  3. Önemli dağılımlar
  4. Beklenti ve varyans
  5. Maksimum Olabilirlik Tahmini (MLE)
  6. Softmax fonksiyonu — LLM çıktı katmanı
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. TEMEL OLASILIK
# ─────────────────────────────────────────────────────────────
# Örnek uzayı Ω, olay A ⊆ Ω
# P(A) ∈ [0, 1],  P(Ω) = 1
#
# Toplama kuralı: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
# Çarpma kuralı: P(A ∩ B) = P(A) * P(B|A)

def temel_olasilik():
    print("=" * 55)
    print("1. TEMEL OLASILIK")
    print("=" * 55)

    # Zar atma simülasyonu
    np.random.seed(42)
    N = 100_000
    zar = np.random.randint(1, 7, size=N)

    # Frekans → olasılık
    for yuz in range(1, 7):
        p_frekans = np.mean(zar == yuz)
        print(f"P(zar={yuz}) ≈ {p_frekans:.4f}  (teorik: {1/6:.4f})")

    # Birleşik olay: P(zar çift VEYA zar > 4)
    cift = (zar % 2 == 0)
    buyuk = (zar > 4)
    p_birlesim = np.mean(cift | buyuk)
    # Teorik: P(çift) = 3/6, P(>4) = 2/6, P(çift VE >4) = P({6}) = 1/6
    # P(çift VEYA >4) = 3/6 + 2/6 - 1/6 = 4/6
    print(f"\nP(çift veya >4) ≈ {p_birlesim:.4f}  (teorik: {4/6:.4f})")


# ─────────────────────────────────────────────────────────────
# 2. KOŞULLU OLASILIK VE BAYES
# ─────────────────────────────────────────────────────────────
# Koşullu olasılık:
#   P(A | B) = P(A ∩ B) / P(B)
#
# Bayes Teoremi:
#   P(A | B) = P(B | A) * P(A) / P(B)
#
# LLM'de Bayes:
#   Dil modeli P(kelime | bağlam) hesaplar.
#   Posterior = Likelihood × Prior / Evidence
#   P(θ | data) ∝ P(data | θ) × P(θ)

def bayes_teoremi():
    print("\n" + "=" * 55)
    print("2. BAYES TEOREMİ")
    print("=" * 55)

    # Klasik örnek: Hastalık testi
    # P(hasta) = 0.001  (prior — hastalık nadirdir)
    # P(test+ | hasta) = 0.99   (sensitivity)
    # P(test+ | sağlıklı) = 0.05 (false positive rate)
    #
    # Test pozitif geldi, gerçekten hasta olma olasılığı?
    # P(hasta | test+) = P(test+ | hasta) * P(hasta) / P(test+)

    p_hasta = 0.001
    p_test_pos_given_hasta = 0.99
    p_test_pos_given_saglikli = 0.05
    p_saglikli = 1 - p_hasta

    # P(test+) = P(test+ | hasta) P(hasta) + P(test+ | sağlıklı) P(sağlıklı)
    p_test_pos = (p_test_pos_given_hasta * p_hasta +
                  p_test_pos_given_saglikli * p_saglikli)

    p_hasta_given_test_pos = (p_test_pos_given_hasta * p_hasta) / p_test_pos

    print(f"P(hasta)            = {p_hasta}")
    print(f"P(test+ | hasta)    = {p_test_pos_given_hasta}")
    print(f"P(test+ | sağlıklı) = {p_test_pos_given_saglikli}")
    print(f"P(test+)            = {p_test_pos:.5f}")
    print(f"P(hasta | test+)    = {p_hasta_given_test_pos:.4f}  ← sadece ~{p_hasta_given_test_pos*100:.1f}%!")
    print("→ Nadir olaylar için prior çok önemli!")


# ─────────────────────────────────────────────────────────────
# 3. ÖNEMLİ DAĞILIMLAR
# ─────────────────────────────────────────────────────────────

def dagilimlar():
    print("\n" + "=" * 55)
    print("3. ÖNEMLİ DAĞILIMLAR")
    print("=" * 55)

    # --- Bernoulli Dağılımı ---
    # P(X=1) = p,  P(X=0) = 1-p
    # E[X] = p,  Var(X) = p(1-p)
    p = 0.3
    samples = np.random.binomial(1, p, size=10000)
    print(f"Bernoulli(p={p}): E[X]≈{samples.mean():.3f} (teorik:{p}), Var≈{samples.var():.3f} (teorik:{p*(1-p):.3f})")

    # --- Normal (Gaussian) Dağılımı ---
    # f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
    # E[X] = μ,  Var(X) = σ²
    # LLM'de: ağırlık başlatma (Xavier, He init) Gaussian kullanır
    mu, sigma = 0.0, 1.0
    samples_normal = np.random.normal(mu, sigma, size=10000)
    print(f"Normal(μ={mu}, σ={sigma}): E[X]≈{samples_normal.mean():.3f}, Std≈{samples_normal.std():.3f}")

    # --- Kategorik Dağılım ---
    # P(X=k) = p_k,  Σ p_k = 1
    # LLM token dağılımı burada! Vocabulary üzerinde dağılım.
    # Softmax çıktısı → kategorik dağılımın parametreleri
    probs = np.array([0.1, 0.3, 0.4, 0.2])   # 4-token vocabulary
    print(f"\nKategorik dağılım: {probs}  (toplam={probs.sum()})")
    samples_cat = np.random.choice(len(probs), p=probs, size=10000)
    print(f"Örnekleme frekansları: {[f'{np.mean(samples_cat==k):.3f}' for k in range(4)]}")


# ─────────────────────────────────────────────────────────────
# 4. BEKLENTI VE VARYANS
# ─────────────────────────────────────────────────────────────
# E[X] = Σ_x x * P(X=x)          (ayrık)
#       = ∫ x * f(x) dx           (sürekli)
#
# Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
# Std(X) = √Var(X)
#
# LLM bağlantısı:
#   - Batch normalization: x_norm = (x - E[x]) / √(Var(x) + ε)
#   - Layer normalization: aynısını feature boyutunda yapar

def beklenti_varyans():
    print("\n" + "=" * 55)
    print("4. BEKLENTI VE VARYANS")
    print("=" * 55)

    # Sıfırdan hesap
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    probs  = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    E_X = np.sum(values * probs)
    E_X2 = np.sum(values**2 * probs)
    Var_X = E_X2 - E_X**2
    Std_X = np.sqrt(Var_X)

    print(f"E[X]    = {E_X:.4f}")
    print(f"Var(X)  = {Var_X:.4f}")
    print(f"Std(X)  = {Std_X:.4f}")

    # Layer Norm'un içi — bu formülü transformer'da göreceğiz
    x = np.array([2.0, 4.0, 3.0, 5.0, 1.0])
    mu = x.mean()
    var = x.var()
    eps = 1e-5
    x_norm = (x - mu) / np.sqrt(var + eps)
    print(f"\nLayer Norm örneği:")
    print(f"  x      = {x}")
    print(f"  μ      = {mu:.4f}")
    print(f"  σ²     = {var:.4f}")
    print(f"  x_norm = {x_norm}")
    print(f"  E[x_norm] ≈ {x_norm.mean():.6f}  (≈ 0)")
    print(f"  Var(x_norm) ≈ {x_norm.var():.6f}  (≈ 1)")


# ─────────────────────────────────────────────────────────────
# 5. MAKSİMUM OLABİLİRLİK TAHMİNİ (MLE)
# ─────────────────────────────────────────────────────────────
# Veri: X = {x_1, ..., x_n}, Model parametreleri: θ
#
# Olabilirlik: L(θ) = P(X | θ) = Π P(x_i | θ)
# Log-olabilirlik: ℓ(θ) = Σ log P(x_i | θ)   (çarpımı toplamaya çevir)
#
# MLE: θ* = argmax_θ ℓ(θ) = argmin_θ -ℓ(θ)
#
# LLM eğitimi = MLE!
# Hedef: θ* = argmax_θ Σ_{t} log P(w_t | w_{<t}; θ)
# Bu da Cross-Entropy Loss'u minimize etmek ile aynıdır.

def mle():
    print("\n" + "=" * 55)
    print("5. MAKSİMUM OLABİLİRLİK TAHMİNİ (MLE)")
    print("=" * 55)

    # Basit örnek: Bernoulli MLE
    # Gözlemlenen veri: n atışta k tane tura
    # L(p) = p^k * (1-p)^{n-k}
    # ℓ(p) = k*log(p) + (n-k)*log(1-p)
    # dℓ/dp = 0 → p* = k/n
    np.random.seed(42)
    p_gercek = 0.7
    n = 1000
    data = np.random.binomial(1, p_gercek, size=n)
    k = data.sum()

    p_mle = k / n  # kapalı form çözüm
    print(f"Gerçek p     = {p_gercek}")
    print(f"MLE tahmini  = {p_mle:.4f}  (k={k}, n={n})")

    # Log-olabilirlik eğrisi
    p_values = np.linspace(0.01, 0.99, 200)
    log_lik = k * np.log(p_values) + (n - k) * np.log(1 - p_values)

    print(f"p={p_mle:.3f}'de log-lik = {k*np.log(p_mle) + (n-k)*np.log(1-p_mle):.2f}  (maksimum)")

    # Gaussian MLE — kapalı form:
    # μ* = (1/n) Σ x_i   (örneklem ortalaması)
    # σ²* = (1/n) Σ (x_i - μ*)²
    x = np.random.normal(loc=3.0, scale=2.0, size=500)
    mu_mle = x.mean()
    sigma2_mle = x.var()
    print(f"\nGaussian MLE: μ*={mu_mle:.3f}, σ²*={sigma2_mle:.3f}")
    print(f"  (gerçek: μ=3.0, σ²=4.0)")


# ─────────────────────────────────────────────────────────────
# 6. SOFTMAX FONKSİYONU
# ─────────────────────────────────────────────────────────────
# LLM'in çıktı katmanı logit vektörü z ∈ R^V üretir (V = vocab size).
# Softmax bunu olasılık dağılımına dönüştürür:
#
#   softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
#
# Özellikler:
#   - Σ softmax(z)_i = 1   (geçerli olasılık dağılımı)
#   - Softmax(z)_i > 0     (her zaman pozitif)
#   - softmax(z + c) = softmax(z)   (sayısal kararlılık için: c = -max(z))
#
# Temperature ile örnekleme:
#   softmax(z / T)
#   T → 0: argmax (greedy)    T = 1: normal     T → ∞: uniform

def softmax():
    print("\n" + "=" * 55)
    print("6. SOFTMAX FONKSİYONU")
    print("=" * 55)

    def softmax_naive(z):
        # Sayısal kararsız — büyük z için overflow!
        return np.exp(z) / np.sum(np.exp(z))

    def softmax_stable(z):
        # Sayısal kararlı: exp(z_i - max(z)) / Σ exp(z_j - max(z))
        z_shift = z - np.max(z)
        exp_z = np.exp(z_shift)
        return exp_z / np.sum(exp_z)

    # Örnek: 5-token vocabulary için logitler
    z = np.array([2.0, 1.0, 0.5, 3.0, -1.0])
    probs = softmax_stable(z)
    print(f"Logitler z:       {z}")
    print(f"Olasılıklar p:    {probs}")
    print(f"Toplam:           {probs.sum():.6f}  (= 1.0 olmalı)")
    print(f"En yüksek token:  index {np.argmax(probs)} (p={probs.max():.4f})")

    # Temperature etkisi
    print("\nTemperature etkisi:")
    for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
        p_T = softmax_stable(z / T)
        print(f"  T={T:.1f}: {p_T}  (max_p={p_T.max():.4f})")

    # Sayısal kararsızlık örneği
    z_büyük = np.array([1000.0, 1001.0, 999.0])
    try:
        p_naive = softmax_naive(z_büyük)
        print(f"\nNaive softmax büyük z: {p_naive}")
    except:
        print("\nNaive softmax OVERFLOW!")
    p_stable = softmax_stable(z_büyük)
    print(f"Stable softmax büyük z: {p_stable}")

    # Top-k ve Top-p (nucleus) sampling
    print("\nTop-k örnekleme (k=2):")
    top_k = 2
    z_topk = z.copy()
    # En büyük k dışındakileri -inf yap
    threshold = np.sort(z_topk)[::-1][top_k - 1]
    z_topk[z_topk < threshold] = -np.inf
    p_topk = softmax_stable(z_topk)
    print(f"  Olasılıklar: {p_topk}")

    print("\nTop-p örnekleme (p=0.9):")
    top_p = 0.9
    sorted_idx = np.argsort(probs)[::-1]
    cumulative = 0
    z_topp = np.full_like(z, -np.inf)
    for idx in sorted_idx:
        z_topp[idx] = z[idx]
        cumulative += probs[idx]
        if cumulative >= top_p:
            break
    p_topp = softmax_stable(z_topp)
    print(f"  Dahil edilen tokenlar: {np.where(z_topp > -np.inf)[0]}")
    print(f"  Olasılıklar: {p_topp}")


if __name__ == "__main__":
    temel_olasilik()
    bayes_teoremi()
    dagilimlar()
    beklenti_varyans()
    mle()
    softmax()
