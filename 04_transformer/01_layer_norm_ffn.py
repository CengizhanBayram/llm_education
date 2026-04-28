"""
=============================================================
MODÜL 4.1 — LAYER NORMALIZATION VE FEED-FORWARD NETWORK
=============================================================

Transformer'ın iki temel alt bileşeni:
  1. Layer Normalization (LN)
  2. Feed-Forward Network (FFN)

Her ikisi de transformer bloğunun vazgeçilmez parçası.

Konular:
  1. Batch Norm vs Layer Norm farkı
  2. Layer Norm matematiksel türetim
  3. Pre-LN vs Post-LN (GPT-2 pre-LN kullanır!)
  4. RMSNorm (LLaMA'da kullanılan)
  5. FFN matematiksel form
  6. PyTorch implementasyon
=============================================================
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. BATCH NORM vs LAYER NORM
# ─────────────────────────────────────────────────────────────
# Batch Norm: batch boyutu üzerinde normalleştir
#   - Her feature için: batch içindeki tüm örneklerin ortalaması/std
#   - Sorun: inference'ta tek örnek var → batch istatistikleri bozulur
#   - Sorun: sekans modellerde farklı uzunluklar
#
# Layer Norm: feature boyutu üzerinde normalleştir
#   - Her örnek için: kendi içindeki tüm feature'ların ortalaması/std
#   - Batch bağımsız → sekans modelleri için ideal
#   - LLM'lerde standart

def batch_vs_layer_norm():
    print("=" * 60)
    print("1. BATCH NORM vs LAYER NORM")
    print("=" * 60)

    batch, seq, d = 3, 4, 8
    X = np.random.randn(batch, seq, d)

    print(f"X shape: {X.shape}  (batch={batch}, seq={seq}, d={d})")

    # Batch Norm: her feature boyutu için batch üzerinde normalleştir
    # Axis = (0, 1) — batch ve seq üzerinden
    X_flat = X.reshape(-1, d)   # (batch*seq, d)
    mu_bn   = X_flat.mean(axis=0)    # (d,)
    std_bn  = X_flat.std(axis=0)     # (d,)
    X_bn = (X_flat - mu_bn) / (std_bn + 1e-5)
    print(f"\nBatch Norm:")
    print(f"  μ örneği (ilk 4 dim): {mu_bn[:4]}")
    print(f"  X_bn mean ≈ {X_bn.mean():.6f}  (≈0)")
    print(f"  X_bn std  ≈ {X_bn.std():.6f}   (≈1)")
    print(f"  Sorun: batch istatistiklerine bağımlı!")

    # Layer Norm: her token için feature boyutunda normalleştir
    mu_ln  = X.mean(axis=-1, keepdims=True)   # (batch, seq, 1)
    std_ln = X.std(axis=-1, keepdims=True)    # (batch, seq, 1)
    X_ln = (X - mu_ln) / (std_ln + 1e-5)
    print(f"\nLayer Norm:")
    print(f"  X_ln mean (per token) ≈ {X_ln.mean(axis=-1).mean():.6f}  (≈0)")
    print(f"  X_ln std  (per token) ≈ {X_ln.std(axis=-1).mean():.6f}   (≈1)")
    print(f"  Batch bağımsız → sekans için mükemmel")


# ─────────────────────────────────────────────────────────────
# 2. LAYER NORM — MATEMATİKSEL TÜRETİM
# ─────────────────────────────────────────────────────────────
# Giriş: x ∈ R^d
#
# Adım 1: Normalleştir
#   x̂ = (x - μ) / √(σ² + ε)
#   μ  = (1/d) Σ x_i
#   σ² = (1/d) Σ (x_i - μ)²
#
# Adım 2: Ölçekle ve kaydır (öğrenilmiş parametreler)
#   y = γ ⊙ x̂ + β
#   γ, β ∈ R^d  (her feature için ayrı)
#   Başlangıç: γ = 1, β = 0  (identity)
#
# Gradyan türetimi:
#   ∂L/∂γ = Σ_i (∂L/∂y_i) * x̂_i
#   ∂L/∂β = Σ_i (∂L/∂y_i)
#   ∂L/∂x: daha karmaşık, PyTorch autograd kullanır

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)    # scale
        self.beta  = np.zeros(d_model)   # shift

    def forward(self, x):
        """x: (..., d_model)"""
        self.x = x
        self.mu    = x.mean(axis=-1, keepdims=True)
        self.var   = x.var(axis=-1, keepdims=True)
        self.x_hat = (x - self.mu) / np.sqrt(self.var + self.eps)
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, dout):
        """Analitik gradyan (karmaşık ama öğretici)"""
        d = self.d_model
        x_hat = self.x_hat
        std_inv = 1.0 / np.sqrt(self.var + self.eps)

        # Gradyan → gamma ve beta
        d_gamma = (dout * x_hat).sum(axis=tuple(range(dout.ndim-1)))
        d_beta  = dout.sum(axis=tuple(range(dout.ndim-1)))

        # Gradyan → x_hat
        d_x_hat = dout * self.gamma

        # Gradyan → x (türetim için Batch Norm backward formülü kullanılır)
        # ∂L/∂x = (1/d*std) * [d * ∂L/∂x̂ - Σ∂L/∂x̂ - x̂ * Σ(∂L/∂x̂ * x̂)]
        d_x = (std_inv / d) * (
            d * d_x_hat
            - d_x_hat.sum(axis=-1, keepdims=True)
            - x_hat * (d_x_hat * x_hat).sum(axis=-1, keepdims=True)
        )
        return d_x, d_gamma, d_beta


def layer_norm_demo():
    print("\n" + "=" * 60)
    print("2. LAYER NORM DEMO")
    print("=" * 60)

    np.random.seed(42)
    d_model = 8
    ln = LayerNorm(d_model)

    # Tek token
    x = np.array([[2.0, 4.0, -1.0, 3.0, 0.5, -2.0, 1.0, 5.0]])
    out = ln.forward(x)

    print(f"Girdi x: {x[0]}")
    print(f"μ = {x.mean(axis=-1)[0]:.4f}")
    print(f"σ² = {x.var(axis=-1)[0]:.4f}")
    print(f"x̂: {np.round(ln.x_hat[0], 4)}")
    print(f"Çıktı y (γ=1,β=0): {np.round(out[0], 4)}")
    print(f"Çıktı mean ≈ {out.mean(axis=-1)[0]:.6f}  (≈0)")
    print(f"Çıktı std  ≈ {out.std(axis=-1)[0]:.6f}   (≈1)")

    # γ, β öğrenilince shape değişir
    ln.gamma = np.array([2.0]*d_model)  # her boyutu 2x büyüt
    ln.beta  = np.array([1.0]*d_model)  # 1 kaydır
    out2 = ln.forward(x)
    print(f"\nγ=2, β=1 ile çıktı: {np.round(out2[0], 4)}")


# ─────────────────────────────────────────────────────────────
# 3. PRE-LN vs POST-LN
# ─────────────────────────────────────────────────────────────
# Orijinal Transformer (2017): Post-LN
#   x' = LN(x + Sublayer(x))
#   → LN son katmandan sonra
#   → Eğitim başında instabil, warm-up gerektirir
#
# GPT-2, GPT-3, LLaMA: Pre-LN
#   x' = x + Sublayer(LN(x))
#   → LN sublayer'dan önce
#   → Daha stabil eğitim, warm-up gerekmeyebilir
#   → Gradyan akışı daha iyi (identity residual path temiz)
#
# Pre-LN avantajı:
#   Residual stream = x (normalize edilmemiş)
#   Gradyan direkt geçer: ∂L/∂x = ∂L/∂x' * 1 (identity) + small

def preln_vs_postln():
    print("\n" + "=" * 60)
    print("3. PRE-LN vs POST-LN")
    print("=" * 60)

    print("""
  POST-LN (Orijinal Transformer 2017):
    x' = LN(x + Sublayer(x))

      x ──────────────────────┐
      │                       │
      └──→ Sublayer(x) ──→ + ──→ LN ──→ x'

  PRE-LN (GPT-2, LLaMA, Mistral):
    x' = x + Sublayer(LN(x))

      x ──────────────────────────────────┐
      │                                   │
      └──→ LN ──→ Sublayer(LN(x)) ──→ + ──→ x'

  Pre-LN'in gradyan avantajı:
    ∂L/∂x = ∂L/∂x' + (küçük sublayer gradyanı)
    → Direkt residual yol sayesinde derin ağlarda gradyan akışı kararlı
    """)


# ─────────────────────────────────────────────────────────────
# 4. RMSNorm — Root Mean Square Layer Normalization
# ─────────────────────────────────────────────────────────────
# Zhang & Sennrich (2019):
#
#   RMSNorm(x) = (x / RMS(x)) * γ
#
#   RMS(x) = √((1/d) Σ x_i²)
#
# Fark: ortalama çıkarma yok! (re-centering yok)
# Sadece re-scaling.
#
# Avantaj: ~%7-10 daha hızlı hesaplanır (μ hesabı yok)
# Pratik sonuç: LayerNorm ile benzer performans
#
# LLaMA, LLaMA-2/3, Mistral, Gemma kullanır.

class RMSNorm:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)

    def forward(self, x):
        # RMS(x) = sqrt(mean(x²))
        rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + self.eps)
        return self.gamma * x / rms


def rmsnorm_demo():
    print("\n" + "=" * 60)
    print("4. RMSNorm vs LayerNorm")
    print("=" * 60)

    np.random.seed(0)
    d = 8
    x = np.random.randn(3, d) * 2 + 1   # sıfır ortalama değil

    ln   = LayerNorm(d)
    rmsn = RMSNorm(d)

    out_ln   = ln.forward(x)
    out_rmsn = rmsn.forward(x)

    print(f"Giriş mean: {x.mean(axis=-1)}")
    print(f"\nLayerNorm çıktı mean: {out_ln.mean(axis=-1)}")
    print(f"RMSNorm çıktı mean:   {out_rmsn.mean(axis=-1)}")
    print("\n→ LayerNorm: mean≈0, std≈1")
    print("→ RMSNorm: mean≠0 (re-centering yok), sadece ölçek")

    # Hız karşılaştırması (konseptsel)
    print("\nHesap karmaşıklığı:")
    print("  LayerNorm: mean (d işlem) + var (d işlem) + norm (d işlem) = 3d")
    print("  RMSNorm:   x² (d işlem) + mean (d işlem) + norm (d işlem) = 3d (ama mean hesabı daha basit)")
    print("  LLaMA tercih: RMSNorm daha az hesap, benzer performans")


# ─────────────────────────────────────────────────────────────
# 5. FEED-FORWARD NETWORK (FFN)
# ─────────────────────────────────────────────────────────────
# Orijinal Transformer:
#   FFN(x) = max(0, xW_1 + b_1) W_2 + b_2    (ReLU)
#
# GPT-2, GPT-3:
#   FFN(x) = GELU(xW_1 + b_1) W_2 + b_2
#
# LLaMA (SwiGLU):
#   FFN(x) = (SiLU(xW_1) ⊙ xW_3) W_2
#
# Boyutlar (GPT-2 convention):
#   d_ff = 4 × d_model
#   W_1 ∈ R^{d_model x d_ff}
#   W_2 ∈ R^{d_ff x d_model}
#
# FFN'nin rolü:
#   - Attention: "hangi token ile bilgi paylaş?" (routing)
#   - FFN: "bu bilgi ile ne yap?" (transformation)
#   - FFN → lookup table gibi: anahtar-değer çiftlerini depolar
#     (Geva et al. 2021 "Transformer Feed-Forward Layers Are Key-Value Memories")

class FFN:
    def __init__(self, d_model, d_ff=None, activation='gelu'):
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation

        scale = np.sqrt(2.0 / (d_model + d_ff))
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

    def gelu(self, x):
        try:
            from scipy.special import erf
            return x * 0.5 * (1 + erf(x / np.sqrt(2)))
        except ImportError:
            # Yaklaşık form
            return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def forward(self, x):
        """x: (..., d_model)"""
        self.x  = x
        self.z1 = x @ self.W1 + self.b1     # (..., d_ff)

        if self.activation == 'gelu':
            self.h = self.gelu(self.z1)
        elif self.activation == 'relu':
            self.h = np.maximum(0, self.z1)

        out = self.h @ self.W2 + self.b2    # (..., d_model)
        return out

    def param_count(self):
        return (self.W1.size + self.b1.size +
                self.W2.size + self.b2.size)


def ffn_demo():
    print("\n" + "=" * 60)
    print("5. FEED-FORWARD NETWORK")
    print("=" * 60)

    d_model = 512
    ffn = FFN(d_model, d_ff=2048)

    print(f"FFN(d={d_model}, d_ff={2048}):")
    print(f"  W1: {ffn.W1.shape}  →  {ffn.W1.size:,} parametre")
    print(f"  W2: {ffn.W2.shape}  →  {ffn.W2.size:,} parametre")
    print(f"  Toplam: {ffn.param_count():,}")

    x = np.random.randn(2, 10, d_model)
    out = ffn.forward(x)
    print(f"\nGiriş: {x.shape} → Çıktı: {out.shape}")

    # GPT-2 small FFN analizi
    print("\n--- GPT-2 FFN Parametre Analizi ---")
    gpt2_cfgs = [
        ("GPT-2 small", 12, 768, 3072),
        ("GPT-2 medium", 24, 1024, 4096),
    ]
    for name, L, d, d_ff in gpt2_cfgs:
        per_layer = 2 * d * d_ff + d_ff + d
        total = L * per_layer
        print(f"  {name}: {L} katman × {per_layer:,} = {total:,} parametre")

    # FFN'nin "memory" rolü
    print("\n--- FFN = KEY-VALUE MEMORY ---")
    print("""
  Geva et al. 2021: FFN katmanları bilgi depolar.
  W1 satırları = "keys" (hangi input pattern?)
  W2 sütunları = "values" (bu pattern için ne döndür?)

  Örnek: Bir FFN nöronu "Paris is the capital of ___" için
  aktif olabilir ve W2'deki ilgili sütun "France" bilgisini içerir.
  → LLM'lerdeki faktüel bilgi büyük ölçüde FFN'de depolanır!
    """)


if __name__ == "__main__":
    np.random.seed(42)
    batch_vs_layer_norm()
    layer_norm_demo()
    preln_vs_postln()
    rmsnorm_demo()
    ffn_demo()
