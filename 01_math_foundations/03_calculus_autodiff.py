"""
=============================================================
MODÜL 1.3 — KALKÜLÜS VE OTOMATİK TÜREVLEme (Calculus & Autodiff)
=============================================================

LLM'lerde neden kalkülüs?
  - Eğitim = loss'u minimize etme = gradyan iniş (gradient descent)
  - Backpropagation = zincir kuralının sistematik uygulaması
  - Her ağırlık için: ∂L/∂W = nasıl değiştirirsem loss düşer?

Konular:
  1. Türev ve kısmi türev
  2. Gradyan vektörü
  3. Zincir kuralı (chain rule)
  4. Jakobian ve Hessian matrisleri
  5. Gradyan iniş
  6. Sıfırdan otomatik türevleme (autograd)
  7. PyTorch autograd
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. TÜREV
# ─────────────────────────────────────────────────────────────
# f'(x) = lim_{h→0} [f(x+h) - f(x)] / h
#
# Geometrik anlam: x noktasındaki teğet eğimi
# Fizik anlamı: anlık değişim hızı

def turev_temel():
    print("=" * 55)
    print("1. TÜREV")
    print("=" * 55)

    # f(x) = x² → f'(x) = 2x  (analitik)
    def f(x):
        return x ** 2

    def f_prime_analitik(x):
        return 2 * x

    def f_prime_sayisal(x, h=1e-5):
        # İleri fark yaklaşımı (forward difference)
        return (f(x + h) - f(x)) / h

    def f_prime_merkezi(x, h=1e-5):
        # Merkezi fark (central difference) — daha doğru: O(h²)
        return (f(x + h) - f(x - h)) / (2 * h)

    x = 3.0
    print(f"f(x)  = x²,  x = {x}")
    print(f"f'(x) analitik  = {f_prime_analitik(x)}")
    print(f"f'(x) ileri fark = {f_prime_sayisal(x):.6f}")
    print(f"f'(x) merkezi    = {f_prime_merkezi(x):.6f}")

    # Önemli türevler (LLM'de sık kullanılan)
    print("\n--- Önemli Türevler ---")
    print("d/dx [x^n]        = n * x^(n-1)")
    print("d/dx [exp(x)]     = exp(x)")
    print("d/dx [log(x)]     = 1/x")
    print("d/dx [sigmoid(x)] = sigmoid(x) * (1 - sigmoid(x))")
    print("d/dx [tanh(x)]    = 1 - tanh²(x)")

    # Sigmoid türevi
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x_vals = np.array([-3.0, 0.0, 1.0, 3.0])
    sig = sigmoid(x_vals)
    sig_prime = sig * (1 - sig)
    print(f"\nSigmoid türevi örneği:")
    for x_v, s, sp in zip(x_vals, sig, sig_prime):
        print(f"  x={x_v:5.1f}: σ(x)={s:.4f}, σ'(x)={sp:.4f}")


# ─────────────────────────────────────────────────────────────
# 2. KISMİ TÜREV VE GRADYAN
# ─────────────────────────────────────────────────────────────
# f: R^n → R için gradyan:
#   ∇f(x) = [∂f/∂x_1, ∂f/∂x_2, ..., ∂f/∂x_n]^T
#
# Gradyan, f'nin en hızlı arttığı yönü gösterir.
# Gradyanın tersi (-∇f) → en hızlı azalma yönü (gradient descent!)

def gradyan():
    print("\n" + "=" * 55)
    print("2. KISMİ TÜREV VE GRADYAN")
    print("=" * 55)

    # f(x, y) = x² + 2xy + y²
    # ∂f/∂x = 2x + 2y
    # ∂f/∂y = 2x + 2y
    # ∇f = [2x+2y, 2x+2y]^T = 2(x+y) * [1, 1]^T

    def f_2d(x, y):
        return x**2 + 2*x*y + y**2  # = (x+y)²

    def grad_f_2d(x, y):
        dfdx = 2*x + 2*y
        dfdy = 2*x + 2*y
        return np.array([dfdx, dfdy])

    x, y = 2.0, 3.0
    g = grad_f_2d(x, y)
    print(f"f(x,y) = (x+y)²,  x={x}, y={y}")
    print(f"f({x},{y}) = {f_2d(x,y)}")
    print(f"∇f({x},{y}) = {g}")

    # Sayısal doğrulama
    h = 1e-5
    grad_x_numerical = (f_2d(x+h, y) - f_2d(x-h, y)) / (2*h)
    grad_y_numerical = (f_2d(x, y+h) - f_2d(x, y-h)) / (2*h)
    print(f"Sayısal ∇f = [{grad_x_numerical:.6f}, {grad_y_numerical:.6f}]")

    # MSE loss gradyanı — Neural network'te temel
    # L(w) = (1/n) Σ (ŷ_i - y_i)²  = (1/n) ||Xw - y||²
    # ∇_w L = (2/n) X^T (Xw - y)
    print("\n--- MSE Loss Gradyanı ---")
    np.random.seed(0)
    n, d = 10, 3
    X = np.random.randn(n, d)
    y = np.random.randn(n)
    w = np.zeros(d)

    y_hat = X @ w
    loss = np.mean((y_hat - y)**2)
    grad_w = (2/n) * X.T @ (y_hat - y)
    print(f"Loss = {loss:.4f},  ||∇L|| = {np.linalg.norm(grad_w):.4f}")


# ─────────────────────────────────────────────────────────────
# 3. ZİNCİR KURALI (Chain Rule)
# ─────────────────────────────────────────────────────────────
# Bileşik fonksiyon: h(x) = f(g(x))
#   h'(x) = f'(g(x)) * g'(x)
#
# Çok değişkenli zincir kuralı:
#   y = f(u),  u = g(x)  →  dy/dx = (dy/du) * (du/dx)
#
# BACKPROPAGATION = zincir kuralının hesap çizgisi üzerinde uygulanması!
# L = loss(softmax(W₂ * relu(W₁ * x)))
# ∂L/∂W₁ = ∂L/∂output * ∂output/∂relu * ∂relu/∂(W₁x) * ∂(W₁x)/∂W₁

def zincir_kurali():
    print("\n" + "=" * 55)
    print("3. ZİNCİR KURALI")
    print("=" * 55)

    # Örnek: L(x) = (sigmoid(x²))²
    # u = x²         → du/dx = 2x
    # v = sigmoid(u) → dv/du = sigmoid(u)(1-sigmoid(u))
    # L = v²         → dL/dv = 2v
    # Zincir: dL/dx = dL/dv * dv/du * du/dx

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def L(x):
        u = x**2
        v = sigmoid(u)
        return v**2

    def dL_dx_analitik(x):
        u = x**2
        v = sigmoid(u)
        dL_dv = 2 * v
        dv_du = v * (1 - v)
        du_dx = 2 * x
        return dL_dv * dv_du * du_dx

    x = 1.5
    h = 1e-6
    dL_dx_sayisal = (L(x+h) - L(x-h)) / (2*h)
    dL_dx_analitik_val = dL_dx_analitik(x)
    print(f"L(x) = sigmoid(x²)²,  x = {x}")
    print(f"dL/dx analitik = {dL_dx_analitik_val:.8f}")
    print(f"dL/dx sayısal  = {dL_dx_sayisal:.8f}")
    print(f"Fark           = {abs(dL_dx_analitik_val - dL_dx_sayisal):.2e}")


# ─────────────────────────────────────────────────────────────
# 4. JAKOBIAN MATRİSİ
# ─────────────────────────────────────────────────────────────
# f: R^n → R^m  (vektörden vektöre fonksiyon)
#
#   J_ij = ∂f_i / ∂x_j   →   J ∈ R^{m x n}
#
# Softmax Jakobiyanı:
#   ∂s_i/∂z_j = s_i(δ_{ij} - s_j)
#   δ_{ij} = 1 if i==j else 0  (Kronecker delta)

def jakobian():
    print("\n" + "=" * 55)
    print("4. JAKOBIAN MATRİSİ — SOFTMAX ÖRNEĞİ")
    print("=" * 55)

    def softmax(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def softmax_jakobian(z):
        s = softmax(z)
        n = len(s)
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    J[i, j] = s[i] * (1 - s[j])  # = s_i(1 - s_i)
                else:
                    J[i, j] = -s[i] * s[j]
        return J

    # Matris formu: J = diag(s) - s @ s^T
    def softmax_jakobian_vektörel(z):
        s = softmax(z)
        return np.diag(s) - np.outer(s, s)

    z = np.array([1.0, 2.0, 0.5])
    J = softmax_jakobian(z)
    J_vec = softmax_jakobian_vektörel(z)
    print(f"z = {z}")
    print(f"softmax(z) = {softmax(z)}")
    print(f"Softmax Jakobiyanı:\n{J}")
    print(f"Vektörel form ile aynı mı? {np.allclose(J, J_vec)}")


# ─────────────────────────────────────────────────────────────
# 5. GRADYAN İNİŞ
# ─────────────────────────────────────────────────────────────
# Parametre güncelleme:
#   θ ← θ - η * ∇_θ L(θ)
#
#   η: öğrenme hızı (learning rate)
#   ∇_θ L: loss'un θ'ya göre gradyanı
#
# Stochastic Gradient Descent (SGD):
#   Her adımda tüm veri yerine mini-batch kullanılır.

def gradyan_inis():
    print("\n" + "=" * 55)
    print("5. GRADYAN İNİŞ")
    print("=" * 55)

    # 1D örnek: f(x) = x⁴ - 4x² + 5  →  minimize et
    # f'(x) = 4x³ - 8x
    def f(x):
        return x**4 - 4*x**2 + 5

    def f_prime(x):
        return 4*x**3 - 8*x

    x = 3.0  # başlangıç noktası
    lr = 0.01
    history = [x]

    for step in range(100):
        grad = f_prime(x)
        x = x - lr * grad
        history.append(x)

    history = np.array(history)
    print(f"Başlangıç: x=3.0, f(x)={f(3.0):.4f}")
    print(f"Son:       x={x:.4f}, f(x)={f(x):.4f}")
    print(f"Minimum noktalar: x=±√2≈±{np.sqrt(2):.4f}")

    # Linear regression ile SGD
    print("\n--- Linear Regression ile SGD ---")
    np.random.seed(42)
    n = 100
    w_gercek = 2.5
    b_gercek = 1.0
    X = np.random.randn(n)
    y = w_gercek * X + b_gercek + 0.1 * np.random.randn(n)

    # Parametreler
    w = 0.0
    b = 0.0
    lr = 0.01
    epochs = 200

    for epoch in range(epochs):
        # Forward pass
        y_hat = w * X + b
        loss = np.mean((y_hat - y)**2)

        # Gradyanlar
        dL_dw = np.mean(2 * (y_hat - y) * X)
        dL_db = np.mean(2 * (y_hat - y))

        # Güncelleme
        w -= lr * dL_dw
        b -= lr * dL_db

    print(f"Tahmin:  w={w:.4f}, b={b:.4f}")
    print(f"Gerçek:  w={w_gercek}, b={b_gercek}")


# ─────────────────────────────────────────────────────────────
# 6. SIFIRDAN OTOMATİK TÜREVLEme (Scalar Autograd)
# ─────────────────────────────────────────────────────────────
# Autograd: her işlemi kaydet → zincir kuralıyla geri git
# Bu, PyTorch'un temel prensibi.

class Scalar:
    """Tek değerli sayı için autograd motor."""

    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')

        def _backward():
            # ∂(a+b)/∂a = 1,  ∂(a+b)/∂b = 1
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')

        def _backward():
            # ∂(a*b)/∂a = b,  ∂(a*b)/∂b = a
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        out = Scalar(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            # ∂(x^n)/∂x = n * x^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Scalar(max(0, self.data), (self,), 'ReLU')

        def _backward():
            # ∂ReLU(x)/∂x = 1 if x > 0 else 0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topolojik sıralama ile tüm düğümleri gez
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Scalar(data={self.data:.4f}, grad={self.grad:.4f})"


def sifirdan_autograd():
    print("\n" + "=" * 55)
    print("6. SIFIRDAN OTOMATİK TÜREVLEme")
    print("=" * 55)

    # L = (x * w + b)²  →  dL/dw ve dL/db
    x = Scalar(2.0)
    w = Scalar(3.0)
    b = Scalar(1.0)

    # Forward pass
    z = x * w + b    # z = 2*3 + 1 = 7
    L = z ** 2        # L = 49

    print(f"Forward: z = {z.data}, L = {L.data}")

    # Backward pass — zincir kuralı otomatik
    L.backward()

    print(f"dL/dw = {w.grad:.4f}  (analitik: 2*(x*w+b)*x = 2*7*2 = {2*7*2})")
    print(f"dL/db = {b.grad:.4f}  (analitik: 2*(x*w+b)*1 = 2*7 = {2*7})")
    print(f"dL/dx = {x.grad:.4f}  (analitik: 2*(x*w+b)*w = 2*7*3 = {2*7*3})")

    # ReLU ile örnek: L = ReLU(x*w)²
    x2 = Scalar(-1.0)
    w2 = Scalar(2.0)
    z2 = (x2 * w2).relu()  # ReLU(-2) = 0
    L2 = z2 ** 2
    L2.backward()
    print(f"\nReLU örneği: z=ReLU({x2.data}*{w2.data})={z2.data}, L={L2.data}")
    print(f"dL/dw2 = {w2.grad}  (sıfır — ReLU x<0'da gradyanı keser!)")


# ─────────────────────────────────────────────────────────────
# 7. PYTORCH AUTOGRAD
# ─────────────────────────────────────────────────────────────

def pytorch_autograd():
    print("\n" + "=" * 55)
    print("7. PYTORCH AUTOGRAD")
    print("=" * 55)

    try:
        import torch

        x = torch.tensor(2.0, requires_grad=True)
        w = torch.tensor(3.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)

        z = x * w + b
        L = z ** 2
        L.backward()

        print(f"PyTorch: L = {L.item()}")
        print(f"dL/dw = {w.grad.item()}  (beklenen: 28)")
        print(f"dL/db = {b.grad.item()}  (beklenen: 14)")
        print(f"dL/dx = {x.grad.item()}  (beklenen: 42)")

        # Matris seviyesi — LLM katmanı örneği
        # y = x @ W,  L = mean(y²)
        torch.manual_seed(0)
        x_mat = torch.randn(3, 4, requires_grad=False)
        W = torch.randn(4, 5, requires_grad=True)

        y = x_mat @ W
        L_mat = (y ** 2).mean()
        L_mat.backward()
        print(f"\nMatris: x@W shape = {y.shape}, dL/dW shape = {W.grad.shape}")

    except ImportError:
        print("PyTorch yüklü değil. 'pip install torch' ile yükleyin.")


if __name__ == "__main__":
    turev_temel()
    gradyan()
    zincir_kurali()
    jakobian()
    gradyan_inis()
    sifirdan_autograd()
    pytorch_autograd()
