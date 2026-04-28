"""
=============================================================
MODÜL 2.1 — PERCEPTRON VE ÇOK KATMANLI ALGILAYICI (MLP)
=============================================================

Sinir ağlarının temel yapı taşı. LLM'deki FFN (Feed-Forward Network)
aslında 2 katmanlı bir MLP'dir.

Konular:
  1. Tek nöron (perceptron) — matematik
  2. İleri geçiş (forward pass)
  3. Çok Katmanlı Algılayıcı (MLP) — NumPy ile sıfırdan
  4. PyTorch ile MLP
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. TEK NÖRON (PERCEPTRON)
# ─────────────────────────────────────────────────────────────
# Giriş: x ∈ R^d
# Ağırlık: w ∈ R^d,  Bias: b ∈ R
#
# Pre-activation (logit): z = w^T x + b = Σ_i w_i x_i + b
# Aktivasyon:             a = f(z)
#
# f: aktivasyon fonksiyonu (sigmoid, ReLU, tanh, ...)
#
# Görsel:
#   x_1 ─── w_1 ─┐
#   x_2 ─── w_2 ─┼──[Σ + b]──[f]──► a
#   x_3 ─── w_3 ─┘

class Perceptron:
    def __init__(self, input_dim):
        # Xavier/Glorot başlatma: Var(w) = 2/(fan_in + fan_out)
        # Tek nöron → fan_out=1
        scale = np.sqrt(2.0 / (input_dim + 1))
        self.w = np.random.randn(input_dim) * scale
        self.b = 0.0

    def forward(self, x):
        # z = w^T x + b
        z = np.dot(self.w, x) + self.b
        # ReLU aktivasyon: a = max(0, z)
        a = np.maximum(0, z)
        return a, z   # aktivasyon ve logit'i döndür

    def __repr__(self):
        return f"Perceptron(dim={len(self.w)}, w={self.w}, b={self.b:.4f})"


def perceptron_demo():
    print("=" * 55)
    print("1. PERCEPTRON")
    print("=" * 55)

    np.random.seed(42)
    neuron = Perceptron(input_dim=3)
    x = np.array([1.0, -2.0, 0.5])

    a, z = neuron.forward(x)
    print(f"x = {x}")
    print(f"w = {neuron.w}")
    print(f"b = {neuron.b:.4f}")
    print(f"z = w·x + b = {z:.4f}")
    print(f"a = ReLU(z) = {a:.4f}")


# ─────────────────────────────────────────────────────────────
# 2. KATMAN (LAYER) — VEKTÖRİZE EDİLMİŞ
# ─────────────────────────────────────────────────────────────
# n nöronlu katman:
#   W ∈ R^{d_out x d_in},  b ∈ R^{d_out}
#   z = Wx + b    →   z ∈ R^{d_out}
#   a = f(z)
#
# Batch işlem için (m örnek):
#   X ∈ R^{m x d_in}
#   Z = X W^T + b   →   Z ∈ R^{m x d_out}   [PyTorch convention]

class DenseLayer:
    def __init__(self, d_in, d_out, activation='relu'):
        # Xavier başlatma
        scale = np.sqrt(2.0 / (d_in + d_out))
        self.W = np.random.randn(d_out, d_in) * scale
        self.b = np.zeros(d_out)
        self.activation = activation
        # Gradient depolaması
        self.dW = None
        self.db = None
        # Cache (backprop için)
        self.cache = {}

    def forward(self, X):
        """X: (batch, d_in) → out: (batch, d_out)"""
        # Z = X @ W^T + b   →   (batch, d_out)
        Z = X @ self.W.T + self.b
        self.cache['X'] = X
        self.cache['Z'] = Z

        if self.activation == 'relu':
            A = np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
        elif self.activation == 'tanh':
            A = np.tanh(Z)
        elif self.activation == 'linear':
            A = Z
        else:
            raise ValueError(f"Bilinmeyen aktivasyon: {self.activation}")

        self.cache['A'] = A
        return A

    def backward(self, dA):
        """dA: upstream gradient (batch, d_out) → dX: (batch, d_in)"""
        Z = self.cache['Z']
        X = self.cache['X']
        batch_size = X.shape[0]

        if self.activation == 'relu':
            dZ = dA * (Z > 0).astype(float)
        elif self.activation == 'sigmoid':
            A = self.cache['A']
            dZ = dA * A * (1 - A)
        elif self.activation == 'tanh':
            A = self.cache['A']
            dZ = dA * (1 - A**2)
        elif self.activation == 'linear':
            dZ = dA

        # dW = dZ^T @ X / batch_size
        self.dW = (dZ.T @ X) / batch_size
        # db = mean(dZ, axis=0)
        self.db = dZ.mean(axis=0)
        # dX = dZ @ W
        dX = dZ @ self.W
        return dX


# ─────────────────────────────────────────────────────────────
# 3. ÇOK KATMANLI ALGILAYICI (MLP) — NUMPY İLE SIFIRDAN
# ─────────────────────────────────────────────────────────────
# Mimari: d_in → [d_hidden]*L → d_out
#
# İleri geçiş:
#   h^(0) = x
#   h^(l) = f(W^(l) h^(l-1) + b^(l))   l=1,...,L
#   ŷ = W^(L+1) h^(L) + b^(L+1)
#
# LLM'deki FFN tam olarak budur:
#   FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2
#   d_ff = 4 * d_model  (GPT-2 convention)

class MLP:
    def __init__(self, layer_dims):
        """
        layer_dims: [d_in, d_h1, d_h2, ..., d_out]
        Son katman linear (sınıflandırma için softmax dışarıda)
        """
        self.layers = []
        for i in range(len(layer_dims) - 1):
            if i < len(layer_dims) - 2:
                act = 'relu'
            else:
                act = 'linear'   # çıktı katmanı
            layer = DenseLayer(layer_dims[i], layer_dims[i+1], activation=act)
            self.layers.append(layer)

    def forward(self, X):
        h = X
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, dL_dout):
        grad = dL_dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    def param_count(self):
        total = 0
        for layer in self.layers:
            total += layer.W.size + layer.b.size
        return total


def mlp_demo():
    print("\n" + "=" * 55)
    print("2. ÇOK KATMANLI ALGILAYICI (MLP)")
    print("=" * 55)

    np.random.seed(42)

    # Mimariler
    mlp_kucuk  = MLP([4, 8, 4, 2])     # 4→8→4→2
    mlp_llm_ff = MLP([512, 2048, 512])  # LLM FFN benzeri (d_model=512, d_ff=2048)

    print(f"Küçük MLP [4→8→4→2]: {mlp_kucuk.param_count():,} parametre")
    print(f"LLM FFN [512→2048→512]: {mlp_llm_ff.param_count():,} parametre")

    # İleri geçiş
    batch_size = 3
    X = np.random.randn(batch_size, 4)
    out = mlp_kucuk.forward(X)
    print(f"\nGiriş shape: {X.shape}")
    print(f"Çıktı shape: {out.shape}")
    print(f"Çıktı:\n{out}")


def mlp_egitim_demo():
    print("\n" + "=" * 55)
    print("3. MLP EĞİTİMİ — İKİLİ SINIFLANDIRMA")
    print("=" * 55)

    # XOR problemi — lineer model çözemez, MLP çözer
    np.random.seed(0)
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)  # XOR

    # Veri genişlet
    N = 1000
    X_train = np.tile(X, (N//4, 1)) + 0.1*np.random.randn(N, 2)
    y_train = np.tile(y, (N//4, 1))

    model = MLP([2, 8, 8, 1])
    lr = 0.1
    epochs = 500

    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def bce_loss(y_hat, y):
        y_hat = np.clip(y_hat, 1e-7, 1-1e-7)
        return -np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

    for epoch in range(epochs):
        # Forward
        logits = model.forward(X_train)
        probs = sigmoid(logits)
        loss = bce_loss(probs, y_train)

        # Backward — BCE + sigmoid gradyanı: dL/dlogits = probs - y
        dL_dlogits = (probs - y_train) / len(y_train)
        model.backward(dL_dlogits)
        model.update(lr)

        if (epoch + 1) % 100 == 0:
            acc = np.mean((probs > 0.5) == y_train)
            print(f"  Epoch {epoch+1:4d}: loss={loss:.4f}, acc={acc:.4f}")

    # Test
    print("\nXOR test:")
    test_out = sigmoid(model.forward(X))
    for xi, yi, pi in zip(X, y, test_out):
        print(f"  {xi} → tahmin={pi[0]:.4f}, gerçek={yi[0]}")


# ─────────────────────────────────────────────────────────────
# 4. PYTORCH İLE MLP
# ─────────────────────────────────────────────────────────────

def pytorch_mlp():
    print("\n" + "=" * 55)
    print("4. PYTORCH İLE MLP")
    print("=" * 55)

    try:
        import torch
        import torch.nn as nn

        # LLM FFN bloğu — tam olarak transformer'daki FFN
        class FFN(nn.Module):
            """
            Transformer FFN:
            FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2
            d_ff = 4 * d_model (GPT-2, GPT-3 convention)
            """
            def __init__(self, d_model, d_ff=None):
                super().__init__()
                d_ff = d_ff or 4 * d_model
                self.fc1 = nn.Linear(d_model, d_ff)
                self.fc2 = nn.Linear(d_ff, d_model)
                self.act = nn.GELU()

            def forward(self, x):
                # x: (batch, seq_len, d_model)
                return self.fc2(self.act(self.fc1(x)))

        d_model = 512
        ffn = FFN(d_model)

        param_count = sum(p.numel() for p in ffn.parameters())
        print(f"FFN(d_model={d_model}): {param_count:,} parametre")

        # GPT-2 small: 12 katman × bu FFN = 12 × {d_model=768, d_ff=3072}
        d_model_gpt2 = 768
        ffn_gpt2 = FFN(d_model_gpt2, d_ff=3072)
        ffn_params = sum(p.numel() for p in ffn_gpt2.parameters())
        print(f"GPT-2 FFN (1 katman): {ffn_params:,} parametre")
        print(f"GPT-2 toplam FFN (12 katman): {ffn_params*12:,} parametre")

        # İleri geçiş
        x = torch.randn(2, 10, d_model)   # (batch=2, seq=10, d_model=512)
        out = ffn(x)
        print(f"\nFFN giriş shape: {x.shape}")
        print(f"FFN çıktı shape: {out.shape}")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    perceptron_demo()
    mlp_demo()
    mlp_egitim_demo()
    pytorch_mlp()
