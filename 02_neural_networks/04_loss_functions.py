"""
=============================================================
MODÜL 2.4 — KAYIP FONKSİYONLARI (Loss Functions)
=============================================================

LLM eğitiminde kullanılan temel loss fonksiyonları:
  - Dil Modeli Eğitimi: Cross-Entropy Loss (Next Token Prediction)
  - RLHF: PPO surrogate loss + KL cezası
  - DPO: Bradley-Terry preference loss

Konular:
  1. MSE (Mean Squared Error)
  2. Binary Cross-Entropy (BCE)
  3. Categorical Cross-Entropy (NLL)  ← LLM'in ana loss'u
  4. Label Smoothing
  5. Loss gradyanlarının türetimi
  6. PyTorch ile loss fonksiyonları
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. MSE — ORTALAMA KARE HATA
# ─────────────────────────────────────────────────────────────
# L_MSE = (1/n) Σ (ŷ_i - y_i)²
#
# Gradyan: ∂L/∂ŷ_i = (2/n)(ŷ_i - y_i)
#
# LLM'de kullanım: Regresyon çıktıları (değer tahmini, embedding regression)

def mse():
    print("=" * 55)
    print("1. MSE — MEAN SQUARED ERROR")
    print("=" * 55)

    def mse_loss(y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def mse_grad(y_hat, y):
        return 2 * (y_hat - y) / len(y)

    y     = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_iyi = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
    y_kötü = np.array([3.0, 1.0, 5.0, 2.0, 0.0])

    print(f"İyi tahmin  MSE: {mse_loss(y_iyi, y):.4f}")
    print(f"Kötü tahmin MSE: {mse_loss(y_kötü, y):.4f}")

    # Özellik: simetrik, büyük hatalara daha fazla ceza
    print(f"\nBüyük hata (5): {mse_loss(np.array([5.0]), np.array([0.0])):.1f}")
    print(f"Küçük hata (1): {mse_loss(np.array([1.0]), np.array([0.0])):.1f}")
    print(f"Büyük/Küçük oranı: 25x  (MSE karesel)")


# ─────────────────────────────────────────────────────────────
# 2. BİNARY CROSS-ENTROPY (BCE)
# ─────────────────────────────────────────────────────────────
# İkili sınıflandırma için:
#   L_BCE = -(1/n) Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
#
# Gradyan: ∂L/∂ŷ_i = -(y_i/ŷ_i - (1-y_i)/(1-ŷ_i)) / n
#
# Sigmoid + BCE → birlikte: ∂L/∂z_i = (σ(z_i) - y_i) / n   (temiz!)

def bce():
    print("\n" + "=" * 55)
    print("2. BINARY CROSS-ENTROPY")
    print("=" * 55)

    def bce_loss(y_hat, y, eps=1e-12):
        y_hat = np.clip(y_hat, eps, 1-eps)
        return -np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

    def bce_grad(y_hat, y, eps=1e-12):
        y_hat = np.clip(y_hat, eps, 1-eps)
        return -(y/y_hat - (1-y)/(1-y_hat)) / len(y)

    y = np.array([1., 0., 1., 0., 1.])

    cases = [
        ("Mükemmel", np.array([0.99, 0.01, 0.99, 0.01, 0.99])),
        ("İyi",      np.array([0.8, 0.2, 0.7, 0.3, 0.9])),
        ("Orta",     np.array([0.6, 0.4, 0.6, 0.4, 0.6])),
        ("Kötü",     np.array([0.2, 0.8, 0.3, 0.7, 0.2])),
    ]
    for name, y_hat in cases:
        loss = bce_loss(y_hat, y)
        acc = np.mean((y_hat > 0.5) == y)
        print(f"  {name:10s}: loss={loss:.4f}, acc={acc:.2f}")

    # Sigmoid + BCE birlikte gradyan
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    z = np.array([2.0, -1.0, 1.5])
    y_bin = np.array([1., 0., 1.])
    p = sigmoid(z)
    grad_sigmoid_bce = (p - y_bin) / len(z)  # temiz form!
    print(f"\nSigmoid+BCE gradyanı (temiz form): {grad_sigmoid_bce}")


# ─────────────────────────────────────────────────────────────
# 3. KATEGORİK CROSS-ENTROPY / NLL
# ─────────────────────────────────────────────────────────────
# Çok sınıflı sınıflandırma:
#   L_CE = -(1/n) Σ_i Σ_k y_{ik} log(q_{ik})
#
# One-hot etiketler için:
#   L_CE = -(1/n) Σ_i log(q_{i, t_i})    t_i: gerçek sınıf indeksi
#
# Bu, LLM'in tam olarak minimize ettiği hedef!
# Her token için: L = -log P(w_t | w_{1:t-1})
#
# Softmax + CE birlikte gradyan (temiz form):
#   ∂L/∂z_k = (q_k - y_k) / n
#   → Tahmin olasılığı gerçekten çıkarılır, normalizasyon yapılır

def categorical_ce():
    print("\n" + "=" * 55)
    print("3. KATEGORİK CROSS-ENTROPY — LLM ANA LOSS")
    print("=" * 55)

    def softmax(z):
        z = np.array(z, dtype=float)
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def ce_loss(logits, targets):
        """
        logits: (n, C) — modelin logitleri
        targets: (n,) — gerçek sınıf indeksleri
        """
        n = len(targets)
        probs = softmax(logits)
        # Her örnek için doğru tokenin log-olasılığını al
        correct_log_probs = np.log(probs[np.arange(n), targets] + 1e-12)
        return -np.mean(correct_log_probs)

    def ce_grad(logits, targets):
        """Softmax + CE gradyanı: q - y"""
        n = len(targets)
        probs = softmax(logits)
        # One-hot'tan çıkar
        probs[np.arange(n), targets] -= 1.0
        return probs / n

    # 4-token vocab, 3 örnek
    vocab_size = 4
    logits = np.array([
        [2.0, 1.0, 0.5, 0.1],  # örnek 1, gerçek token: 0
        [0.1, 0.2, 3.0, 0.1],  # örnek 2, gerçek token: 2
        [0.5, 2.5, 0.3, 0.2],  # örnek 3, gerçek token: 1
    ])
    targets = np.array([0, 2, 1])

    loss = ce_loss(logits, targets)
    grad = ce_grad(logits.copy(), targets)

    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {np.exp(loss):.4f}")
    print(f"Gradyan (logitlere göre):\n{grad}")
    print(f"\nGradyan yorumu: q - one_hot(y)")
    print("  → Pozitif: bu tokeni daha az tahmin et")
    print("  → Negatif: bu tokeni daha fazla tahmin et")

    # LLM eğitim senaryosu
    print("\n--- LLM EĞİTİM SENARYOSU ---")
    print("Girdi: 'The cat sat on the'")
    print("Hedef token: 'mat' (index 5432)")
    vocab = 50257   # GPT-2 vocab
    model_logits = np.random.randn(vocab) * 0.1
    # Modeli biraz doğru yönde ayarla
    model_logits[5432] = 3.0

    probs = softmax(model_logits)
    loss_single = -np.log(probs[5432] + 1e-12)
    print(f"P('mat') = {probs[5432]:.6f}")
    print(f"Loss = -log P('mat') = {loss_single:.4f}")
    print(f"Perplexity = {np.exp(loss_single):.2f}")


# ─────────────────────────────────────────────────────────────
# 4. LABEL SMOOTHING
# ─────────────────────────────────────────────────────────────
# One-hot yerine yumuşatılmış etiket:
#   y_smooth_k = (1 - ε) * y_k + ε/K
#
# ε = 0.1: gerçek sınıfa 0.9, diğerlerine 0.1/K
#
# Faydaları:
#   - Overconfidence önleme: model p=1 yerine p=0.9 hedefler
#   - Regularizasyon etkisi → genelleme iyileşir
#   - "Attention Is All You Need" paper'ında kullanıldı: ε=0.1
#
# LLM eğitiminde:
#   - GPT-2/3: label smoothing yok
#   - T5: ε=0.1
#   - Modern LLM'ler: genellikle kullanılmıyor (büyük modeller zaten regularize)

def label_smoothing():
    print("\n" + "=" * 55)
    print("4. LABEL SMOOTHING")
    print("=" * 55)

    def smooth_labels(targets, n_classes, eps=0.1):
        n = len(targets)
        y_smooth = np.full((n, n_classes), eps / n_classes)
        y_smooth[np.arange(n), targets] = 1.0 - eps + eps / n_classes
        return y_smooth

    def smooth_ce_loss(logits, y_smooth):
        def softmax(z):
            z = z - z.max(axis=-1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=-1, keepdims=True)
        probs = softmax(logits)
        return -np.mean(np.sum(y_smooth * np.log(probs + 1e-12), axis=-1))

    logits = np.array([
        [3.0, 1.0, 0.5, 0.1],
        [0.1, 4.0, 0.3, 0.1],
    ])
    targets = np.array([0, 1])
    n_classes = 4

    # One-hot labels
    y_onehot = np.zeros((len(targets), n_classes))
    y_onehot[np.arange(len(targets)), targets] = 1.0

    # Smooth labels
    y_smooth = smooth_labels(targets, n_classes, eps=0.1)

    # Normal CE
    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        return np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)

    probs = softmax(logits)
    ce_normal = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=-1))
    ce_smooth  = smooth_ce_loss(logits, y_smooth)

    print(f"One-hot CE:   {ce_normal:.4f}")
    print(f"Smoothed CE:  {ce_smooth:.4f}")
    print(f"\nSmoothed labels (ε=0.1, K=4):")
    for i, (t, y) in enumerate(zip(targets, y_smooth)):
        print(f"  örnek {i} (t={t}): {y}")
    print("→ Sıfır olasılık yok, her sınıfa küçük katkı var")


# ─────────────────────────────────────────────────────────────
# 5. LOSS GRADYANLARININ TÜRETİMİ
# ─────────────────────────────────────────────────────────────

def loss_grad_turetimi():
    print("\n" + "=" * 55)
    print("5. LOSS GRADYANLARININ TÜRETİMİ")
    print("=" * 55)

    print("Softmax + CE gradyanı türetimi:")
    print("""
  L = -Σ_k y_k log(q_k)     q_k = softmax(z)_k = exp(z_k) / Σ exp(z_j)

  ∂L/∂z_i = -Σ_k y_k * (∂log q_k / ∂z_i)
           = -Σ_k y_k * (1/q_k) * (∂q_k/∂z_i)

  Softmax Jacobian: ∂q_k/∂z_i = q_k(δ_{ki} - q_i)

  ∂L/∂z_i = -Σ_k y_k * (1/q_k) * q_k(δ_{ki} - q_i)
           = -Σ_k y_k(δ_{ki} - q_i)
           = -y_i + q_i * Σ_k y_k

  One-hot için Σ y_k = 1:
           = q_i - y_i   ← ÇOK TEMİZ!
    """)

    # Sayısal doğrulama
    def softmax(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    z = np.array([1.0, 3.0, 0.5, 2.0])
    t = 1  # true class

    q = softmax(z)
    y = np.zeros(len(z))
    y[t] = 1.0

    # Analitik gradyan: q - y
    grad_analytic = q - y

    # Sayısal gradyan
    h = 1e-6
    ce_loss = lambda z: -np.log(softmax(z)[t] + 1e-12)
    grad_numerical = np.array([(ce_loss(z + h*np.eye(len(z))[i]) -
                                 ce_loss(z - h*np.eye(len(z))[i])) / (2*h)
                                for i in range(len(z))])

    print(f"Analitik (q-y): {grad_analytic}")
    print(f"Sayısal:        {grad_numerical}")
    print(f"Max fark:       {np.max(np.abs(grad_analytic - grad_numerical)):.2e}")


# ─────────────────────────────────────────────────────────────
# 6. PYTORCH İLE LOSS FONKSİYONLARI
# ─────────────────────────────────────────────────────────────

def pytorch_losses():
    print("\n" + "=" * 55)
    print("6. PYTORCH İLE LOSS FONKSİYONLARI")
    print("=" * 55)

    try:
        import torch
        import torch.nn as nn

        # CrossEntropyLoss = LogSoftmax + NLLLoss
        # logits → softmax → log → negatif → ortalama
        ce_loss = nn.CrossEntropyLoss()
        ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)

        # LLM senaryosu: batch=2, seq=5, vocab=100
        batch, seq, vocab = 2, 5, 100
        logits = torch.randn(batch * seq, vocab)    # (10, 100)
        targets = torch.randint(0, vocab, (batch * seq,))  # (10,)

        loss = ce_loss(logits, targets)
        loss_smooth = ce_smooth(logits, targets)

        print(f"Normal CE loss:  {loss.item():.4f}")
        print(f"Smoothed CE loss: {loss_smooth.item():.4f}")
        print(f"Perplexity: {torch.exp(loss).item():.4f}")

        # Gradyan kontrolü
        logits.requires_grad_(True)
        loss = ce_loss(logits, targets)
        loss.backward()
        print(f"Logit gradyanı shape: {logits.grad.shape}")
        print(f"Gradyan normu: {logits.grad.norm().item():.4f}")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    mse()
    bce()
    categorical_ce()
    label_smoothing()
    loss_grad_turetimi()
    pytorch_losses()
