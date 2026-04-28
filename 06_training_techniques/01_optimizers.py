"""
=============================================================
MODÜL 6.1 — OPTİMİZATÖRLER
=============================================================

LLM eğitiminde optimizer:
  - GPT-2/3: Adam (β1=0.9, β2=0.95)
  - LLaMA: AdamW (β1=0.9, β2=0.95, wd=0.1)
  - Chinchilla, Gemini: AdamW
  - Muon (2024): Nesterov + Orthogonalization — yeni trend

Konular:
  1. SGD ve Momentum
  2. RMSprop
  3. Adam — sıfırdan türetim
  4. AdamW — weight decay düzeltmesi
  5. Optimizer karşılaştırması
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. SGD VE MOMENTUM
# ─────────────────────────────────────────────────────────────
# Vanilla SGD:
#   θ ← θ - η * g_t
#
# SGD + Momentum:
#   v_t = β * v_{t-1} + g_t
#   θ   ← θ - η * v_t
#
# β: momentum katsayısı (genellikle 0.9)
# v: hız vektörü — geçmiş gradyanların ağırlıklı ortalaması
#
# Sezgi: topun yokuştan inmesi gibi — geçmiş hız birikir

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params   = params
        self.lr       = lr
        self.momentum = momentum
        self.velocity = {id(p): np.zeros_like(v) for p, v in params.items()}

    def step(self, grads):
        for name, param in self.params.items():
            g = grads[name]
            if self.momentum > 0:
                self.velocity[id(param)] = (self.momentum * self.velocity[id(param)] + g)
                param -= self.lr * self.velocity[id(param)]
            else:
                param -= self.lr * g


# ─────────────────────────────────────────────────────────────
# 2. ADAM — Adaptive Moment Estimation
# ─────────────────────────────────────────────────────────────
# Kingma & Ba (2015)
#
# Her parametre için adaptif öğrenme hızı.
#
# Güncelleme kuralı:
#   m_t = β1 * m_{t-1} + (1 - β1) * g_t       [1. moment = ortalama]
#   v_t = β2 * v_{t-1} + (1 - β2) * g_t²       [2. moment = varyans]
#
# Bias düzeltmesi (başlangıçta m,v sıfır olduğu için):
#   m̂_t = m_t / (1 - β1^t)
#   v̂_t = v_t / (1 - β2^t)
#
# Güncelleme:
#   θ_t = θ_{t-1} - η * m̂_t / (√v̂_t + ε)
#
# Önerilen değerler:
#   β1 = 0.9,  β2 = 0.999,  ε = 1e-8,  η = 1e-3
#   LLM'de: β1=0.9, β2=0.95 (daha az 2. moment ataletı)
#
# Adam'ın L2 weight decay ile ilişkisi:
#   Adam + L2 = θ ← θ - η * (m̂/(√v̂+ε) + λθ)
#   AMA: λθ terimi adaptif ölçekleniyor → istenmeyen etki
#   ÇÖZÜM: AdamW

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}   # 1. moment (ortalama)
        self.v = {}   # 2. moment (kare ortalama)
        self.t = 0    # adım sayacı

    def step(self, params, grads):
        self.t += 1
        for name, theta in params.items():
            g = grads[name]

            # Moment kayıtlarını başlat
            if name not in self.m:
                self.m[name] = np.zeros_like(theta)
                self.v[name] = np.zeros_like(theta)

            # 1. moment güncelle
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g

            # 2. moment güncelle
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g**2

            # Bias düzeltmesi
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Güncelleme
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ─────────────────────────────────────────────────────────────
# 3. AdamW — Adam + Düzgün Weight Decay
# ─────────────────────────────────────────────────────────────
# Loshchilov & Hutter (2017) "Decoupled Weight Decay Regularization"
#
# Problem: Adam'da L2 regularizasyon adaptif güncellemeyle karışır
#   Adam+L2: g_t' = g_t + λθ_t  (gradient'e eklenir, sonra adaptif ölçeklenir)
#   → Ağır parametreler daha fazla, hafif parametreler daha az decay
#
# AdamW: weight decay'i gradient'ten AYRI uygula:
#   θ_t = θ_{t-1} - η * (m̂_t/(√v̂_t+ε) + λ * θ_{t-1})
#                              ↑ Adam update  ↑ Doğrudan decay
#
# → Her parametre eşit oranda decay görür (adaptif ölçeklemeden bağımsız)
# → LLM'de standart: AdamW(β1=0.9, β2=0.95, wd=0.1)
#   weight_decay=0.1 → büyük ağırlıklara daha sert ceza

class AdamW:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads, no_decay=None):
        """
        no_decay: weight decay uygulanmayacak parametre isimleri (bias, LayerNorm weight)
        """
        self.t += 1
        no_decay = no_decay or []

        for name, theta in params.items():
            g = grads[name]

            if name not in self.m:
                self.m[name] = np.zeros_like(theta)
                self.v[name] = np.zeros_like(theta)

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g**2

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Adam update
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Weight decay (ayrı, sadece weight matrislere)
            if name not in no_decay:
                update += self.lr * self.wd * theta

            params[name] -= update


# ─────────────────────────────────────────────────────────────
# 4. OPTİMİZATÖR KARŞILAŞTIRMA DENEYİ
# ─────────────────────────────────────────────────────────────

def optimizer_karsilastirma():
    print("=" * 60)
    print("OPTİMİZATÖR KARŞILAŞTIRMASI")
    print("=" * 60)

    # Rosenbrock fonksiyonu — klasik optimizasyon test fonksiyonu
    # f(x,y) = (a-x)² + b(y-x²)²   (a=1, b=100)
    # Global minimum: (1, 1)
    def f(x, y, a=1, b=100):
        return (a - x)**2 + b * (y - x**2)**2

    def grad_f(x, y, a=1, b=100):
        dfdx = -2*(a - x) - 4*b*(y - x**2)*x
        dfdy = 2*b*(y - x**2)
        return np.array([dfdx, dfdy])

    np.random.seed(0)

    optimizers_data = {}
    for opt_name in ["SGD", "Momentum", "Adam", "AdamW"]:
        x = np.array([-1.5, 1.5], dtype=float)

        if opt_name == "SGD":
            lr = 0.001
        elif opt_name == "Momentum":
            lr = 0.001
            v = np.zeros(2)
            beta = 0.9
        elif opt_name in ("Adam", "AdamW"):
            lr = 0.01
            m_a = np.zeros(2)
            v_a = np.zeros(2)
            beta1, beta2, eps_a = 0.9, 0.999, 1e-8

        history = [x.copy()]
        loss_hist = []

        for t in range(1, 1001):
            g = grad_f(x[0], x[1])
            loss_hist.append(f(x[0], x[1]))

            if opt_name == "SGD":
                x -= lr * g

            elif opt_name == "Momentum":
                v = beta * v + g
                x -= lr * v

            elif opt_name == "Adam":
                m_a = 0.9 * m_a + 0.1 * g
                v_a = 0.999 * v_a + 0.001 * g**2
                m_hat = m_a / (1 - 0.9**t)
                v_hat = v_a / (1 - 0.999**t)
                x -= lr * m_hat / (np.sqrt(v_hat) + eps_a)

            elif opt_name == "AdamW":
                m_a = 0.9 * m_a + 0.1 * g
                v_a = 0.999 * v_a + 0.001 * g**2
                m_hat = m_a / (1 - 0.9**t)
                v_hat = v_a / (1 - 0.999**t)
                x -= lr * (m_hat / (np.sqrt(v_hat) + eps_a) + 0.01 * x)

        optimizers_data[opt_name] = {
            'final': x.copy(),
            'final_loss': loss_hist[-1],
            'converged': f(x[0], x[1]) < 0.01,
        }

    print(f"Rosenbrock fonksiyonu: min (1,1), f=0")
    print(f"\n{'Optimizer':12s}  {'Final (x,y)':>22s}  {'Loss':>12s}  {'Yakınsadı?':>12s}")
    print("-" * 65)
    for name, data in optimizers_data.items():
        xy = f"({data['final'][0]:.3f}, {data['final'][1]:.3f})"
        print(f"{name:12s}  {xy:>22s}  {data['final_loss']:>12.6f}  {'✓' if data['converged'] else '✗':>12s}")


def adam_bias_duzeltmesi():
    print("\n" + "=" * 60)
    print("ADAM BIAS DÜZELTMESİ NEDEN GEREKLİ?")
    print("=" * 60)

    print("""
  Problem: t=1'de m_0 = v_0 = 0 (sıfır başlatma)
    m_1 = 0.9 * 0 + 0.1 * g_1 = 0.1 * g_1  (çok küçük!)
    v_1 = 0.999 * 0 + 0.001 * g_1² = 0.001 * g_1²  (çok küçük!)

  Bias düzeltmesi olmadan güncelleme:
    θ -= η * (0.1 * g) / √(0.001 * g²) = η * 0.1/√0.001 * sign(g) ≈ 3η * sign(g)
    → Başlangıçta büyük güncellemeler!

  Bias düzeltmesi ile:
    m̂_1 = m_1 / (1 - 0.9^1) = 0.1*g / 0.1 = g
    v̂_1 = v_1 / (1 - 0.999^1) = 0.001*g² / 0.001 = g²
    θ -= η * g / √(g²) = η * sign(g)  ← doğru!

  t büyüdükçe bias düzeltmesi küçülür (1 - β^t → 1).
    """)

    # Sayısal gösterim
    g = np.array([2.0])
    beta1, beta2 = 0.9, 0.999
    m, v = np.zeros(1), np.zeros(1)

    print("Adım  | m       | v       | m̂(bias düz.) | v̂(bias düz.)")
    print("-" * 62)
    for t in range(1, 6):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        print(f"  {t}   | {m[0]:.5f} | {v[0]:.7f} | {m_hat[0]:.5f}      | {v_hat[0]:.5f}")


def weight_decay_aciklama():
    print("\n" + "=" * 60)
    print("WEIGHT DECAY VE L2 REGULARIZASYON")
    print("=" * 60)

    print("""
  L2 Regularizasyon:
    Loss_total = Loss_data + (λ/2) * ||θ||²
    → Gradyan: g_total = g_data + λ * θ
    → Adam'da: g_total adaptif ölçeklenir → weight decay "erimez"

  AdamW (Decoupled):
    θ ← θ - η * (Adam_step(g_data) + λ * θ)
    → Weight decay, gradient güncellemesinden AYRI
    → Her parametre aynı oranda küçülür

  Hangi parametrelere weight decay?
    - YES: Attention W_Q, W_K, W_V, W_O, FFN W1, W2
    - NO: Bias terms, LayerNorm γ, β, embedding weights

  LLaMA-2 değerleri:
    weight_decay = 0.1  (ağırlıklar yavaşça sıfıra çekiliyor)
    lr = 3e-4 (7B), 1.5e-4 (70B)
    β1=0.9, β2=0.95
    """)


def pytorch_optimizerlar():
    print("\n" + "=" * 60)
    print("PYTORCH OPTİMİZATÖRLER")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        model = nn.Linear(10, 5)

        # AdamW — parametre grupları
        # Weight decay sadece weight matrislere, bias'a değil
        decay_params = [p for n, p in model.named_parameters() if 'weight' in n]
        no_decay_params = [p for n, p in model.named_parameters() if 'bias' in n]

        optimizer = torch.optim.AdamW([
            {'params': decay_params,    'weight_decay': 0.1},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

        print(f"AdamW parametre grupları:")
        print(f"  Weight decay: {sum(p.numel() for p in decay_params):,} parametre")
        print(f"  No decay:     {sum(p.numel() for p in no_decay_params):,} parametre")

        # Basit güncelleme adımı
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        print(f"\nGüncelleme başarılı. Loss: {loss.item():.4f}")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    optimizer_karsilastirma()
    adam_bias_duzeltmesi()
    weight_decay_aciklama()
    pytorch_optimizerlar()
