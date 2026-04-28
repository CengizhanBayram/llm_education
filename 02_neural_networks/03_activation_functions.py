"""
=============================================================
MODÜL 2.3 — AKTİVASYON FONKSİYONLARI
=============================================================

LLM'lerin kullandığı aktivasyonlar:
  - GPT-1/2: GELU
  - GPT-3: GELU
  - LLaMA: SiLU (Swish)
  - LLaMA-2/3, Mistral: SwiGLU  (= SiLU × Linear gate)
  - PaLM: SwiGLU

Neden aktivasyon lazım?
  - Aktivasyon olmadan: tüm katmanlar tek bir lineer dönüşüme indirgenir!
  - W_3(W_2(W_1 x)) = (W_3 W_2 W_1) x  — sadece tek bir matris çarpımı
  - Aktivasyon → non-linearity → karmaşık fonksiyonlar öğrenilebilir

Konular:
  1. Sigmoid, Tanh
  2. ReLU ve varyantları (Leaky, ELU)
  3. GELU — GPT'de kullanılan
  4. SiLU/Swish
  5. SwiGLU — LLaMA'da kullanılan
  6. Aktivasyon karşılaştırması
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. SİGMOID VE TANH — TARİHSEL
# ─────────────────────────────────────────────────────────────
# sigmoid(x) = 1 / (1 + exp(-x))   ∈ (0, 1)
# tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))   ∈ (-1, 1)
#
# Sorunlar:
#   - Vanishing gradient: türev max 0.25 (sigmoid), 1.0 (tanh x=0'da)
#   - Sigmoid çıktısı 0-centered değil → gradyan güncellemesi yavaşlar
#   - Hesaplamalı olarak pahalı (exp)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)   # max = 0.25

def tanh_fn(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2   # max = 1.0 (x=0'da)


# ─────────────────────────────────────────────────────────────
# 2. ReLU VE VARYANTLARı
# ─────────────────────────────────────────────────────────────
# ReLU(x) = max(0, x)
#
# Türev: 1 if x > 0 else 0
#
# Avantajlar:
#   - Basit ve hızlı
#   - x > 0'da gradyan = 1 → vanishing gradient yok
#   - Seyrek aktivasyon (x<0 → 0) → hesaplamalı verimli
#
# Sorunlar:
#   - "Dying ReLU": x < 0 → her zaman 0, nöron ölüyor
#   - Gradient x<0'da = 0, bir kez negatif olursa güncelleme yok
#
# Leaky ReLU: f(x) = max(αx, x), α = 0.01
# ELU: f(x) = x if x>0 else α(exp(x)-1)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_prime(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_prime(x, alpha=1.0):
    return np.where(x > 0, 1.0, elu(x, alpha) + alpha)


# ─────────────────────────────────────────────────────────────
# 3. GELU — Gaussian Error Linear Unit
# ─────────────────────────────────────────────────────────────
# GELU(x) = x * Φ(x)
#
# Burada Φ(x) = CDF of N(0,1) = P(X ≤ x)  [standart normal CDF]
#
# Sezgi: Giriş x'i, x'in bir Gaussian gürültüsünden büyük olma
# olasılığı ile ağırlıklandır.
# → Stochastic regularization gibi davranır.
#
# Tam form:
#   GELU(x) = x * Φ(x) = x * 0.5 * [1 + erf(x / √2)]
#
# Yaklaşık form (GPT-2'de kullanılan):
#   GELU(x) ≈ 0.5x * [1 + tanh(√(2/π) * (x + 0.044715 x³))]
#
# Özellikler:
#   - ReLU'dan farklı: x < 0'da sıfıra zorla DEĞİL, küçük negatif değer
#   - Türev her yerde tanımlı (smooth)
#   - LLM'lerde standart hale geldi: BERT, GPT-2/3

from scipy.special import erf as scipy_erf

def gelu(x):
    # Tam form
    return x * 0.5 * (1 + scipy_erf(x / np.sqrt(2)))

def gelu_approx(x):
    # Yaklaşık form (tanh tabanlı) — GPT-2'de bu kullanılır
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_prime(x):
    # Türev: GELU'(x) = Φ(x) + x * φ(x)
    # φ(x) = PDF of N(0,1) = exp(-x²/2) / √(2π)
    phi = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)   # PDF
    PHI = 0.5 * (1 + scipy_erf(x / np.sqrt(2)))        # CDF
    return PHI + x * phi


# ─────────────────────────────────────────────────────────────
# 4. SiLU / SWISH
# ─────────────────────────────────────────────────────────────
# SiLU(x) = x * sigmoid(x)
#          = x / (1 + exp(-x))
#
# Swish (Ramachandran et al. 2017): f(x) = x * sigmoid(βx)
# β=1 → SiLU
#
# Özellikler:
#   - GELU'ya çok benzer (pratikte neredeyse aynı performans)
#   - Hesaplaması daha basit (erf gerektirmiyor)
#   - LLaMA, Mistral kullanır
#
# Türev:
#   SiLU'(x) = sigmoid(x) + x * sigmoid'(x)
#            = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
#            = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))

def silu(x):
    return x * sigmoid(x)

def silu_prime(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)  # = sigmoid(x)(1 + x(1-sigmoid(x)))


# ─────────────────────────────────────────────────────────────
# 5. SwiGLU — LLaMA / Mistral / PaLM
# ─────────────────────────────────────────────────────────────
# SwiGLU (Noam Shazeer, 2020):
#
# SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
#
# Burada ⊙ element-wise çarpmadır.
#
# LLaMA FFN'i:
#   FFN_SwiGLU(x) = (SiLU(x W_1) ⊙ (x W_3)) W_2
#
# Neden 3 matris?  W_1, W_2, W_3
# → Gating mekanizması: W_3 x'i "kapı" olarak kullanır
# → Gate * Value = hangi bilgiyi geçireceğini öğrenir
#
# NOT: d_ff = 2/3 * 4 * d_model  (3 matris yüzünden parametre sayısını dengele)

def swiglu(x, W1, W3, W2):
    """
    x:  (..., d_model)
    W1: (d_model, d_ff)
    W3: (d_model, d_ff)
    W2: (d_ff, d_model)
    """
    gate  = silu(x @ W1)   # SiLU gate
    value = x @ W3          # Linear value
    hidden = gate * value   # Element-wise gate
    return hidden @ W2      # Project back


def aktivasyon_karsilastirma():
    print("=" * 55)
    print("AKTİVASYON FONKSİYONLARI KARŞILAŞTIRMASI")
    print("=" * 55)

    try:
        from scipy.special import erf as scipy_erf
    except ImportError:
        print("scipy gerekli: pip install scipy")
        return

    x_vals = np.linspace(-3, 3, 7)

    print(f"\n{'x':>6}  {'ReLU':>8}  {'GELU':>8}  {'SiLU':>8}  {'tanh':>8}")
    print("-" * 50)
    for x in x_vals:
        r = relu(float(x))
        g = gelu(float(x))
        s = silu(float(x))
        t = tanh_fn(float(x))
        print(f"{x:>6.2f}  {r:>8.4f}  {g:>8.4f}  {s:>8.4f}  {t:>8.4f}")

    # Türev karşılaştırması
    print(f"\n{'x':>6}  {'ReLU\'':>8}  {'GELU\'':>8}  {'SiLU\'':>8}")
    print("-" * 38)
    for x in x_vals:
        r = relu_prime(float(x))
        g = gelu_prime(float(x))
        s = silu_prime(float(x))
        print(f"{x:>6.2f}  {r:>8.4f}  {g:>8.4f}  {s:>8.4f}")

    # Önemli farklar
    print("\n--- ÖNEMLİ FARKLAR ---")
    print("ReLU(-1) = 0.0000  (keskin, türev=0)")
    print(f"GELU(-1) = {gelu(-1.0):.4f}  (küçük negatif, smooth)")
    print(f"SiLU(-1) = {silu(-1.0):.4f}  (küçük negatif, smooth)")
    print("→ GELU ve SiLU, negatif bölgede küçük sinyal geçirir")
    print("→ Bu, gradyan akışını iyileştirir")


def swiglu_demo():
    print("\n" + "=" * 55)
    print("SwiGLU DEMO (LLaMA FFN)")
    print("=" * 55)

    np.random.seed(42)
    d_model = 512
    # LLaMA: d_ff = int(8/3 * d_model) sonraki 256'nın katına yuvarla
    d_ff = int(8/3 * d_model)
    d_ff = (d_ff + 255) // 256 * 256   # 256'nın katına yuvarla → 1408

    W1 = np.random.randn(d_model, d_ff) * 0.01
    W3 = np.random.randn(d_model, d_ff) * 0.01
    W2 = np.random.randn(d_ff, d_model) * 0.01

    batch, seq = 2, 10
    x = np.random.randn(batch, seq, d_model)

    out = swiglu(x, W1, W3, W2)
    print(f"Giriş shape: {x.shape}")
    print(f"W1, W3: {W1.shape},  W2: {W2.shape}")
    print(f"Çıktı shape: {out.shape}")

    params_swiglu = W1.size + W2.size + W3.size
    params_gelu = 2 * (d_model * (4*d_model))  # 2 matris, d_ff=4*d_model
    print(f"\nSwiGLU parametreleri (3 matris): {params_swiglu:,}")
    print(f"GELU FFN parametreleri (2 matris, d_ff=4*d): {params_gelu:,}")
    print(f"Fark: SwiGLU ~%{(params_swiglu/params_gelu-1)*100:.1f}")


def pytorch_aktivasyonlar():
    print("\n" + "=" * 55)
    print("PYTORCH AKTİVASYONLAR")
    print("=" * 55)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        x = torch.randn(3, 4)

        activations = {
            'ReLU':  nn.ReLU(),
            'GELU':  nn.GELU(),
            'SiLU':  nn.SiLU(),
            'Tanh':  nn.Tanh(),
        }

        for name, fn in activations.items():
            out = fn(x)
            print(f"{name:8s}: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")

        # SwiGLU PyTorch implementasyonu
        class SwiGLU(nn.Module):
            def __init__(self, d_model, d_ff):
                super().__init__()
                self.w1 = nn.Linear(d_model, d_ff, bias=False)
                self.w3 = nn.Linear(d_model, d_ff, bias=False)
                self.w2 = nn.Linear(d_ff, d_model, bias=False)

            def forward(self, x):
                return self.w2(F.silu(self.w1(x)) * self.w3(x))

        d_model = 512
        d_ff    = 1408
        ffn = SwiGLU(d_model, d_ff)
        x_torch = torch.randn(2, 10, d_model)
        out = ffn(x_torch)
        print(f"\nSwiGLU: {x_torch.shape} → {out.shape}")
        print(f"Parametre: {sum(p.numel() for p in ffn.parameters()):,}")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    try:
        from scipy.special import erf as scipy_erf
        aktivasyon_karsilastirma()
    except ImportError:
        print("scipy gerekli: pip install scipy")

    swiglu_demo()
    pytorch_aktivasyonlar()
