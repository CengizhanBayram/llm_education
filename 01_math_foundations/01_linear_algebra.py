"""
=============================================================
MODÜL 1.1 — LİNEER CEBİR (Linear Algebra for LLMs)
=============================================================

LLM'lerde neden lineer cebir?
  - Token embedding'ler → vektörler (R^d)
  - Attention ağırlıkları → matrisler (R^{n x n})
  - Weight matrisler → lineer dönüşümler (R^{d_in x d_out})
  - SVD → weight decomposition, LoRA gibi PEFT yöntemlerinin temeli

Konular:
  1. Vektörler ve temel işlemler
  2. Matris çarpımı
  3. Transpoz, iz (trace), determinant
  4. Öz değer / öz vektör (eigendecomposition)
  5. Tekil Değer Ayrışımı (SVD)
  6. LLM bağlantıları
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. VEKTÖRLER
# ─────────────────────────────────────────────────────────────
# Bir vektör v ∈ R^d, d boyutlu bir sayı dizisidir.
# LLM'de: her token bir d-boyutlu embedding vektörüdür.
#
#   v = [v_1, v_2, ..., v_d]^T

def vektor_temel():
    print("=" * 50)
    print("1. VEKTÖRLER")
    print("=" * 50)

    v = np.array([1.0, 2.0, 3.0])   # R^3 vektörü
    u = np.array([4.0, 5.0, 6.0])

    # Toplama: (v + u)_i = v_i + u_i
    print(f"v + u = {v + u}")

    # Skalar çarpım: (αv)_i = α * v_i
    alpha = 2.0
    print(f"2v    = {alpha * v}")

    # L2 normu: ||v||_2 = sqrt(Σ v_i^2)
    # Geometrik anlam: vektörün uzunluğu
    norm_v = np.sqrt(np.sum(v ** 2))
    print(f"||v||_2 = {norm_v:.4f}  (np: {np.linalg.norm(v):.4f})")

    # Birim vektör: v̂ = v / ||v||
    v_hat = v / np.linalg.norm(v)
    print(f"v̂ (unit vector) = {v_hat}")
    print(f"||v̂|| = {np.linalg.norm(v_hat):.4f}  (her zaman 1)")

    # Nokta çarpımı (dot product): v · u = Σ v_i * u_i
    # Geometrik anlam: v · u = ||v|| ||u|| cos(θ)
    # LLM'de: attention'da Q ve K vektörlerinin benzerliğini ölçer
    dot = np.dot(v, u)
    cos_theta = dot / (np.linalg.norm(v) * np.linalg.norm(u))
    print(f"v · u = {dot}")
    print(f"cos(θ) = {cos_theta:.4f}  →  θ = {np.degrees(np.arccos(cos_theta)):.2f}°")

    # Cosine similarity — LLM'de embedding benzerliği için kritik
    # sim(v, u) = (v · u) / (||v|| * ||u||) ∈ [-1, 1]
    print(f"Cosine similarity = {cos_theta:.4f}")


# ─────────────────────────────────────────────────────────────
# 2. MATRİS ÇARPIMI
# ─────────────────────────────────────────────────────────────
# A ∈ R^{m x k},  B ∈ R^{k x n}  →  C = AB ∈ R^{m x n}
#
# C_{ij} = Σ_{l=1}^{k} A_{il} * B_{lj}
#
# LLM'de:
#   - Linear layer: y = xW + b   (x: (batch, d_in), W: (d_in, d_out))
#   - Attention skoru: QK^T       (Q: (n, d_k), K: (n, d_k))

def matris_carpimi():
    print("\n" + "=" * 50)
    print("2. MATRİS ÇARPIMI")
    print("=" * 50)

    A = np.array([[1, 2, 3],
                  [4, 5, 6]])   # shape (2, 3)

    B = np.array([[7,  8],
                  [9,  10],
                  [11, 12]])   # shape (3, 2)

    # Elle hesap: C_{00} = 1*7 + 2*9 + 3*11 = 58
    C = A @ B   # shape (2, 2)
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"C = A @ B:\n{C}")

    # Önemli özellikler:
    #   AB ≠ BA (genel olarak, matris çarpımı değişmez değil)
    #   (AB)C = A(BC) (birleşme özelliği)
    #   (AB)^T = B^T A^T (transpoz)

    # Batch matris çarpımı — LLM'de sıkça kullanılır
    # x: (batch=2, seq_len=3, d_model=4)
    # W: (d_model=4, d_out=5)
    # y = x @ W → (2, 3, 5)
    x = np.random.randn(2, 3, 4)
    W = np.random.randn(4, 5)
    y = x @ W
    print(f"\nBatch matmul: {x.shape} @ {W.shape} → {y.shape}")


# ─────────────────────────────────────────────────────────────
# 3. TRANSPOZ, İZ, DETERMİNANT
# ─────────────────────────────────────────────────────────────

def transpoz_iz_determinant():
    print("\n" + "=" * 50)
    print("3. TRANSPOZ, İZ, DETERMİNANT")
    print("=" * 50)

    A = np.array([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])

    # Transpoz: A^T_{ij} = A_{ji}
    print(f"A^T:\n{A.T}")

    # İz (trace): tr(A) = Σ A_{ii}  — sadece kare matrisler
    # Öz değerlerin toplamına eşittir
    print(f"tr(A) = {np.trace(A)}")

    B = np.array([[3., 1.],
                  [2., 4.]])

    # Determinant: det(A) — matrisin "hacim ölçekleme faktörü"
    # det = 0 → matris tekil (singular), terslenemiyor
    print(f"det(B) = {np.linalg.det(B):.4f}")

    # Matris tersi: A^{-1} öyle ki A * A^{-1} = I
    B_inv = np.linalg.inv(B)
    print(f"B^{{-1}}:\n{B_inv}")
    print(f"B @ B^{{-1}} ≈ I:\n{np.round(B @ B_inv, 10)}")


# ─────────────────────────────────────────────────────────────
# 4. ÖZ DEĞER AYRIŞIMI (Eigendecomposition)
# ─────────────────────────────────────────────────────────────
# Av = λv
#   v: öz vektör (eigenvector) — A tarafından sadece ölçeklenen yön
#   λ: öz değer (eigenvalue) — ölçekleme faktörü
#
# Simetrik matris A = Q Λ Q^T
#   Q: öz vektör matrisi (ortogonal)
#   Λ: öz değerlerin diyagonal matrisi
#
# LLM bağlantısı:
#   - Dikkat matrisinin öz değerleri → bilgi akışının "güçlü yönleri"
#   - PCA → öz ayrışım ile boyut indirgeme

def oz_deger_ayrisiimi():
    print("\n" + "=" * 50)
    print("4. ÖZ DEĞER AYRIŞIMI")
    print("=" * 50)

    # Simetrik pozitif tanımlı matris oluştur
    A = np.array([[4., 2.],
                  [2., 3.]])

    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"Öz değerler (λ): {eigenvalues}")
    print(f"Öz vektörler (sütunlar):\n{eigenvectors}")

    # Doğrulama: Av = λv
    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        v = eigenvectors[:, i]
        Av = A @ v
        lv = lam * v
        print(f"λ_{i}={lam:.3f}: Av={Av}, λv={lv}  eşit mi? {np.allclose(Av, lv)}")

    # Yeniden inşa: A = Q Λ Q^{-1}
    Q = eigenvectors
    Lambda = np.diag(eigenvalues)
    A_reconstructed = Q @ Lambda @ np.linalg.inv(Q)
    print(f"Yeniden inşa doğru mu? {np.allclose(A, A_reconstructed)}")


# ─────────────────────────────────────────────────────────────
# 5. TEKİL DEĞER AYRIŞIMI (SVD)
# ─────────────────────────────────────────────────────────────
# Her A ∈ R^{m x n} için:
#   A = U Σ V^T
#
#   U ∈ R^{m x m}: sol tekil vektörler (ortogonal)
#   Σ ∈ R^{m x n}: diyagonal, tekil değerler (σ_1 ≥ σ_2 ≥ ... ≥ 0)
#   V ∈ R^{n x n}: sağ tekil vektörler (ortogonal)
#
# En önemli kullanım:
#   Düşük-rank yaklaşım: A ≈ U_k Σ_k V_k^T  (sadece k en büyük tekil değer)
#
# LLM bağlantısı:
#   LoRA (Low-Rank Adaptation): ΔW = BA
#   A ∈ R^{d x r}, B ∈ R^{r x d},  r << d
#   → Milyarlarca parametreli modeli az parametre ile ince ayar yapmak için

def svd_ayrisiimi():
    print("\n" + "=" * 50)
    print("5. TEKİL DEĞER AYRIŞIMI (SVD)")
    print("=" * 50)

    np.random.seed(42)
    A = np.random.randn(4, 6)   # 4x6 matris

    U, sigma, Vt = np.linalg.svd(A, full_matrices=True)
    print(f"A shape: {A.shape}")
    print(f"U shape: {U.shape}, sigma shape: {sigma.shape}, Vt shape: {Vt.shape}")
    print(f"Tekil değerler (σ): {sigma}")

    # Tam yeniden inşa
    Sigma = np.zeros_like(A)
    Sigma[:len(sigma), :len(sigma)] = np.diag(sigma)
    A_reconstructed = U @ Sigma @ Vt
    print(f"Tam SVD yeniden inşa hatası: {np.max(np.abs(A - A_reconstructed)):.2e}")

    # Düşük-rank yaklaşım: sadece k=2 tekil değer kullan
    # A_k = U[:, :k] @ diag(sigma[:k]) @ Vt[:k, :]
    k = 2
    A_lowrank = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
    frobenius_error = np.linalg.norm(A - A_lowrank, 'fro')
    print(f"\nDüşük-rank (k={k}) yaklaşım Frobenius hatası: {frobenius_error:.4f}")
    print(f"Orijinal nükleer norm:  {np.sum(sigma):.4f}")
    print(f"Yakalanan bilgi: {np.sum(sigma[:k]) / np.sum(sigma) * 100:.1f}%")

    # LoRA bağlantısı:
    # W (d=512, d=512) → 512*512 = 262,144 parametre
    # LoRA: A (512, r=8) + B (r=8, 512) → 512*8*2 = 8,192 parametre  (%3.1!)
    d, r = 512, 8
    lora_params = d * r * 2
    full_params = d * d
    print(f"\nLoRA örneği: d={d}, r={r}")
    print(f"  Tam ağırlık: {full_params:,} parametre")
    print(f"  LoRA ağırlık: {lora_params:,} parametre ({lora_params/full_params*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# 6. LLM BAĞLANTISI — EMBEDDING UZAYI
# ─────────────────────────────────────────────────────────────

def llm_baglantisi():
    print("\n" + "=" * 50)
    print("6. LLM BAĞLANTISI — EMBEDDING UZAYI")
    print("=" * 50)

    # GPT-2 small: vocab_size=50257, d_model=768
    # Embedding tablosu: E ∈ R^{50257 x 768}
    # Her token → 768-boyutlu vektör
    vocab_size, d_model = 100, 8   # küçük örnek

    np.random.seed(0)
    E = np.random.randn(vocab_size, d_model)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)  # normalize et

    # Token 5 ve token 7 ne kadar benzer?
    token_a, token_b, token_c = 5, 7, 42
    sim_ab = np.dot(E[token_a], E[token_b])
    sim_ac = np.dot(E[token_a], E[token_c])
    print(f"Cosine sim(token_{token_a}, token_{token_b}) = {sim_ab:.4f}")
    print(f"Cosine sim(token_{token_a}, token_{token_c}) = {sim_ac:.4f}")

    # Linear layer (projection): y = xW
    # x: tek token embedding (d_model=8)
    # W: projeksiyon (d_model=8, d_out=4)
    x = E[token_a]
    W = np.random.randn(d_model, 4) * 0.1
    y = x @ W
    print(f"\nLinear projeksiyon: {x.shape} @ {W.shape} → {y.shape}")
    print(f"Çıktı: {y}")


if __name__ == "__main__":
    vektor_temel()
    matris_carpimi()
    transpoz_iz_determinant()
    oz_deger_ayrisiimi()
    svd_ayrisiimi()
    llm_baglantisi()
