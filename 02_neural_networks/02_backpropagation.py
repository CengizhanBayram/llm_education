"""
=============================================================
MODÜL 2.2 — GERİ YAYILIM (Backpropagation)
=============================================================

Backpropagation = zincir kuralının hesap çizgisi (computational graph)
üzerinde verimli uygulanması.

Bu dosyada:
  1. Hesap çizgisi kavramı
  2. Elle backprop türetimi
  3. NumPy ile tam 2-katmanlı ağ backprop
  4. Sayısal gradyan kontrolü (gradient check)
=============================================================
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. HESAP ÇİZGİSİ VE LOKAL GRADYANLAR
# ─────────────────────────────────────────────────────────────
# Her işlem (node) şunları bilir:
#   - Forward: z = f(a, b, ...)
#   - Backward: ∂z/∂a, ∂z/∂b, ...  (lokal gradyanlar)
#
# Zincir kuralı:
#   ∂L/∂a = (∂L/∂z) * (∂z/∂a)
#   upstream_grad × local_grad

class AddGate:
    """z = a + b"""
    def forward(self, a, b):
        self.a, self.b = a, b
        return a + b

    def backward(self, dz):
        # ∂(a+b)/∂a = 1,  ∂(a+b)/∂b = 1
        return dz * 1, dz * 1


class MulGate:
    """z = a * b"""
    def forward(self, a, b):
        self.a, self.b = a, b
        return a * b

    def backward(self, dz):
        # ∂(a*b)/∂a = b,  ∂(a*b)/∂b = a
        return dz * self.b, dz * self.a


class SigmoidGate:
    """z = sigmoid(x) = 1 / (1 + exp(-x))"""
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dz):
        # ∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x))
        return dz * self.out * (1 - self.out)


class ReLUGate:
    """z = max(0, x)"""
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dz):
        # ∂ReLU(x)/∂x = 1 if x > 0 else 0
        return dz * (self.x > 0)


def hesap_cizgisi_demo():
    print("=" * 55)
    print("1. HESAP ÇİZGİSİ ÖRNEK: f(x,y,z) = (x+y)*z")
    print("=" * 55)

    # f(x, y, z) = (x + y) * z
    # ∂f/∂x = z,  ∂f/∂y = z,  ∂f/∂z = x+y
    x, y, z = 3.0, -2.0, 4.0

    add = AddGate()
    mul = MulGate()

    # Forward
    q = add.forward(x, y)    # q = x + y = 1
    f = mul.forward(q, z)    # f = q * z = 4

    print(f"Forward: q = {q}, f = {f}")

    # Backward — L = f, dL/df = 1
    dL_df = 1.0
    dL_dq, dL_dz = mul.backward(dL_df)
    dL_dx, dL_dy  = add.backward(dL_dq)

    print(f"Backward:")
    print(f"  dL/dz = {dL_dz}  (analitik: x+y = {x+y})")
    print(f"  dL/dq = {dL_dq}  (analitik: z = {z})")
    print(f"  dL/dx = {dL_dx}  (analitik: z = {z})")
    print(f"  dL/dy = {dL_dy}  (analitik: z = {z})")


# ─────────────────────────────────────────────────────────────
# 2. 2-KATMANLI AĞ BACKPROP — ELLE TÜRETİM
# ─────────────────────────────────────────────────────────────
# Mimari: x → Linear1 → ReLU → Linear2 → Loss
#
# Forward:
#   z1 = W1 x + b1       (d_h x 1)
#   h  = ReLU(z1)        (d_h x 1)
#   z2 = W2 h + b2       (d_out x 1)
#   L  = MSE(z2, y)      (scalar)
#
# Backward (zincir kuralı):
#   dL/dz2 = 2*(z2 - y)/n
#   dL/dW2 = dL/dz2 * h^T
#   dL/db2 = dL/dz2
#   dL/dh  = W2^T * dL/dz2
#   dL/dz1 = dL/dh * 1[z1 > 0]    (ReLU türevi)
#   dL/dW1 = dL/dz1 * x^T
#   dL/db1 = dL/dz1

class TwoLayerNet:
    def __init__(self, d_in, d_hidden, d_out):
        scale1 = np.sqrt(2.0 / d_in)
        scale2 = np.sqrt(2.0 / d_hidden)
        self.W1 = np.random.randn(d_hidden, d_in) * scale1
        self.b1 = np.zeros(d_hidden)
        self.W2 = np.random.randn(d_out, d_hidden) * scale2
        self.b2 = np.zeros(d_out)

    def forward(self, X):
        """X: (n, d_in)"""
        self.X   = X
        self.Z1  = X @ self.W1.T + self.b1    # (n, d_h)
        self.H   = np.maximum(0, self.Z1)      # (n, d_h)  ReLU
        self.Z2  = self.H @ self.W2.T + self.b2  # (n, d_out)
        return self.Z2

    def loss(self, Z2, y):
        diff = Z2 - y
        return 0.5 * np.mean(diff ** 2)

    def backward(self, y):
        n = self.X.shape[0]

        # dL/dZ2: MSE türevi (0.5 * mean faktörüyle)
        dZ2 = (self.Z2 - y) / n               # (n, d_out)

        # dL/dW2, dL/db2
        dW2 = dZ2.T @ self.H                  # (d_out, d_h)
        db2 = dZ2.sum(axis=0)                  # (d_out,)

        # dL/dH = dZ2 @ W2
        dH  = dZ2 @ self.W2                    # (n, d_h)

        # dL/dZ1: ReLU türevi = dH * 1[Z1 > 0]
        dZ1 = dH * (self.Z1 > 0)              # (n, d_h)

        # dL/dW1, dL/db1
        dW1 = dZ1.T @ self.X                  # (d_h, d_in)
        db1 = dZ1.sum(axis=0)                  # (d_h,)

        self.grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return self.grads

    def update(self, lr):
        self.W1 -= lr * self.grads['W1']
        self.b1 -= lr * self.grads['b1']
        self.W2 -= lr * self.grads['W2']
        self.b2 -= lr * self.grads['b2']

    def get_params(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}


def two_layer_backprop():
    print("\n" + "=" * 55)
    print("2. 2-KATMANLI AĞ BACKPROP")
    print("=" * 55)

    np.random.seed(42)
    net = TwoLayerNet(d_in=4, d_hidden=8, d_out=2)

    # Örnek veri
    X = np.random.randn(5, 4)
    y = np.random.randn(5, 2)

    # Forward
    Z2 = net.forward(X)
    L = net.loss(Z2, y)
    print(f"Loss: {L:.6f}")

    # Backward
    grads = net.backward(y)
    for name, g in grads.items():
        print(f"  d{name} shape: {g.shape}, norm: {np.linalg.norm(g):.6f}")


# ─────────────────────────────────────────────────────────────
# 3. SAYISAL GRADYAN KONTROLÜ (Gradient Check)
# ─────────────────────────────────────────────────────────────
# Backprop'un doğruluğunu doğrula:
#   Sayısal: ∂L/∂w_i ≈ [L(w + hε_i) - L(w - hε_i)] / 2h
#   Analitik: backprop'tan gelen gradyan
#
# Relative error = ||grad_analitik - grad_sayisal|| / ||grad_analitik + grad_sayisal||
# Kabul edilebilir: < 1e-5

def gradient_check():
    print("\n" + "=" * 55)
    print("3. SAYISAL GRADYAN KONTROLÜ")
    print("=" * 55)

    np.random.seed(0)
    net = TwoLayerNet(d_in=3, d_hidden=4, d_out=2)
    X = np.random.randn(10, 3)
    y = np.random.randn(10, 2)

    # Analitik gradyanlar
    net.forward(X)
    analytic_grads = net.backward(y)

    # Sayısal gradyanlar (merkezi fark)
    h = 1e-5
    params = net.get_params()
    numerical_grads = {}

    for param_name in ['W1', 'b1', 'W2', 'b2']:
        param = params[param_name]
        num_grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            original = param[idx]

            param[idx] = original + h
            L_plus = net.loss(net.forward(X), y)

            param[idx] = original - h
            L_minus = net.loss(net.forward(X), y)

            param[idx] = original
            num_grad[idx] = (L_plus - L_minus) / (2 * h)
            it.iternext()

        numerical_grads[param_name] = num_grad

    # Karşılaştır
    print(f"{'Parametre':6s}  {'Rel. Error':>15s}  {'Sonuç':>10s}")
    print("-" * 38)
    for name in ['W1', 'b1', 'W2', 'b2']:
        a = analytic_grads[name].flatten()
        n = numerical_grads[name].flatten()
        rel_err = np.linalg.norm(a - n) / (np.linalg.norm(a + n) + 1e-12)
        ok = "PASS ✓" if rel_err < 1e-4 else "FAIL ✗"
        print(f"{name:6s}  {rel_err:>15.2e}  {ok:>10s}")


# ─────────────────────────────────────────────────────────────
# 4. VANİSHİNG / EXPLODING GRADIENT
# ─────────────────────────────────────────────────────────────
# Derin ağlarda backprop sırasında gradyanlar:
#   - Exponential olarak küçülür → Vanishing Gradient
#   - Exponential olarak büyür → Exploding Gradient
#
# Nedenler:
#   - Sigmoid/tanh: çıktıları sıkıştırır, türevleri < 1
#   - Derin ağlar: zincir kuralında küçük sayıların çarpımı → 0
#
# Çözümler (LLM'de kullanılanlar):
#   - ReLU / GELU aktivasyon
#   - Residual connections (skip connections) — ∂L/∂x = ∂L/∂(x + F(x)) = ∂L/∂F + I
#   - Layer Normalization
#   - Gradient clipping

def vanishing_gradient_demo():
    print("\n" + "=" * 55)
    print("4. VANİSHİNG / EXPLODING GRADIENT")
    print("=" * 55)

    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    # Sigmoid derin ağ — gradyan küçülmesi
    print("Sigmoid aktivasyonlu derin ağ:")
    x = 0.5
    grad = 1.0  # dL/d_son_katman = 1
    for depth in range(1, 11):
        s = sigmoid(x)
        local_grad = s * (1 - s)   # sigmoid türevi, max 0.25 (x=0'da)
        grad *= local_grad
        if depth in [1, 2, 5, 10]:
            print(f"  Katman {depth:2d}: gradyan = {grad:.2e}  ← {'VANISHING!' if grad < 1e-4 else 'OK'}")

    print("\nReLU aktivasyonlu ağ:")
    x_relu = 0.5
    grad_relu = 1.0
    for depth in range(1, 11):
        local_grad = 1.0 if x_relu > 0 else 0.0   # ReLU türevi
        grad_relu *= local_grad
        if depth in [1, 2, 5, 10]:
            print(f"  Katman {depth:2d}: gradyan = {grad_relu:.2e}  ← {'ZERO (dead ReLU)' if grad_relu == 0 else 'ALIVE'}")

    # Residual connection etkisi
    print("\nResidual connection (skip) ile gradyan:")
    # ∂L/∂x = ∂L/∂(x + F(x)) * (1 + ∂F/∂x)
    # = upstream_grad * (1 + small_number)  → gradyan asla 0 olmaz!
    grad_res = 1.0
    F_prime = 0.1  # küçük F türevi varsay
    for depth in range(1, 11):
        grad_res *= (1 + F_prime)  # her katmanda 1 ekleniyor
        if depth in [1, 2, 5, 10]:
            print(f"  Katman {depth:2d}: gradyan = {grad_res:.4f}  ← stabil")


# ─────────────────────────────────────────────────────────────
# 5. PYTORCH İLE BACKPROP DOĞRULAMA
# ─────────────────────────────────────────────────────────────

def pytorch_backprop():
    print("\n" + "=" * 55)
    print("5. PYTORCH İLE BACKPROP DOĞRULAMA")
    print("=" * 55)

    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(42)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.fc2 = nn.Linear(8, 2)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        net = Net()
        X = torch.randn(5, 4)
        y = torch.randn(5, 2)

        out = net(X)
        loss = 0.5 * ((out - y) ** 2).mean()
        loss.backward()

        print(f"Loss: {loss.item():.6f}")
        for name, param in net.named_parameters():
            print(f"  d{name} shape: {param.grad.shape}, norm: {param.grad.norm().item():.6f}")

        # Gradyan akışını görselleştir
        print("\nGradyan normu (katman başına):")
        for name, param in net.named_parameters():
            print(f"  {name}: {param.grad.norm().item():.6f}")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    hesap_cizgisi_demo()
    two_layer_backprop()
    gradient_check()
    vanishing_gradient_demo()
    pytorch_backprop()
