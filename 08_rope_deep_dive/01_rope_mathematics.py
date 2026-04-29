"""
=============================================================
MODÜL 8.1 — RoPE MATEMATİĞİ — DERİNLEMESİNE
=============================================================

RoPE (Rotary Position Embedding)
Su et al. 2021 — "RoFormer: Enhanced Transformer with Rotary PE"

Kullanımı:
  GPT-NeoX, LLaMA 1/2/3, Mistral, Falcon, Qwen, Phi, Gemma, ...
  → Günümüzde decoder-only LLM'lerin fiili standardı.

Bu dosyada:
  1. 2D rotasyon temeli (kompleks sayı yorumu)
  2. RoPE'nin tam matematiksel türetimi
  3. Görecelilik özelliğinin kanıtı
  4. Frekans analizi (θ_i değerleri)
  5. NumPy ile sıfırdan tam implementasyon
  6. PyTorch ile etkin implementasyon
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. 2D ROTASYON VE KOMPLEKS SAYI TABANLI BAKIŞ
# ─────────────────────────────────────────────────────────────
# Kartezyen 2D döndürme matrisi:
#
#   R(θ) = [ cos θ  -sin θ ]
#           [ sin θ   cos θ ]
#
# x ∈ R² vektörünü θ açısıyla döndürür:
#   x' = R(θ) x
#
# Kompleks sayı gösterimi:
#   x = (x₀, x₁) ↔ z = x₀ + i·x₁   (kompleks sayı)
#   Döndürme: z' = z · e^{iθ} = z · (cosθ + i·sinθ)
#
# RoPE'nin fikri:
#   Her token'a konum m atanır.
#   Boyutları çiftlere böl: (x₀, x₁), (x₂, x₃), ..., (x_{d-2}, x_{d-1})
#   Her çifti kendi frekansıyla döndür:
#   (x_{2i}, x_{2i+1}) → döndür(θ_i · m)

def rotasyon_temeli():
    print("=" * 65)
    print("1. 2D ROTASYON VE KOMPLEKS SAYI YORUMU")
    print("=" * 65)

    # 2D vektörü θ açısıyla döndür
    def rotate2d(x0, x1, theta):
        """
        [ x0' ]   [ cos θ  -sin θ ] [ x0 ]
        [ x1' ] = [ sin θ   cos θ ] [ x1 ]
        """
        x0_rot = x0 * np.cos(theta) - x1 * np.sin(theta)
        x1_rot = x0 * np.sin(theta) + x1 * np.cos(theta)
        return x0_rot, x1_rot

    # Örnek: birim vektör (1, 0) 45° döndür
    x0, x1 = 1.0, 0.0
    theta = np.pi / 4   # 45°
    x0r, x1r = rotate2d(x0, x1, theta)
    print(f"Vektör: ({x0}, {x1})")
    print(f"45° döndürülmüş: ({x0r:.4f}, {x1r:.4f})")
    print(f"Beklenen: ({np.cos(theta):.4f}, {np.sin(theta):.4f})  ✓")

    # Norm korunuyor mu?
    norm_orig = np.sqrt(x0**2 + x1**2)
    norm_rot  = np.sqrt(x0r**2 + x1r**2)
    print(f"Orijinal norm: {norm_orig:.4f},  Döndürülmüş norm: {norm_rot:.4f}")
    print("→ Döndürme norm-koruyucu (isometric dönüşüm) ✓")

    # Kompleks sayı yorumu
    print("\nKompleks sayı yorumu:")
    z = complex(x0, x1)
    e_itheta = complex(np.cos(theta), np.sin(theta))   # e^{iθ}
    z_rot = z * e_itheta
    print(f"  z = {z.real} + {z.imag}i")
    print(f"  e^{{iπ/4}} = {e_itheta.real:.4f} + {e_itheta.imag:.4f}i")
    print(f"  z · e^{{iθ}} = {z_rot.real:.4f} + {z_rot.imag:.4f}i  ← aynı sonuç ✓")


# ─────────────────────────────────────────────────────────────
# 2. RoPE TAM FORMÜLÜ
# ─────────────────────────────────────────────────────────────
# d boyutlu vektör q, m konumundaki token için:
#
# θ_i = 10000^{-2i/d},   i = 0, 1, ..., d/2 - 1
#
# RoPE(q, m) = R_m q
#
# Döndürme matrisi R_m (seyrek):
#   R_m = blok-diyagonal matris, her blok 2×2:
#
#   R_m = diag(R_m^0, R_m^1, ..., R_m^{d/2-1})
#
#   R_m^i = [ cos(m·θ_i)  -sin(m·θ_i) ]
#            [ sin(m·θ_i)   cos(m·θ_i) ]
#
# Vektörel form (verimli hesap):
#   q' = q ⊙ cos(m·θ) + rotate_half(q) ⊙ sin(m·θ)
#
#   rotate_half: (q₀, q₁, q₂, q₃, ...) → (-q₁, q₀, -q₃, q₂, ...)

def rope_formul():
    print("\n" + "=" * 65)
    print("2. RoPE TAM FORMÜLÜ")
    print("=" * 65)

    d = 8   # boyut (sade gösterim için küçük)
    base = 10000.0

    # Frekanslar: θ_i = base^{-2i/d}
    i = np.arange(d // 2)
    theta = base ** (-2 * i / d)   # (d/2,)

    print(f"d = {d},  base = {base}")
    print(f"\nFrekanslar θ_i = {base}^(-2i/{d}):")
    for idx, th in enumerate(theta):
        print(f"  θ_{idx} = {base}^(-{2*idx}/{d}) = {th:.6f}  "
              f"(periyot = 2π/θ = {2*np.pi/th:.2f} token)")

    # m konumundaki q vektörü için rotasyon
    m = 5   # konum
    q = np.random.randn(d)
    print(f"\nq = {np.round(q, 4)}")

    # Yöntem 1: Matris çarpımı (seyrek blok-diyagonal)
    def rope_matrix(q, m, theta):
        d = len(q)
        q_rot = np.zeros(d)
        for idx in range(d // 2):
            angle = m * theta[idx]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            q_rot[2*idx]   = cos_a * q[2*idx]   - sin_a * q[2*idx+1]
            q_rot[2*idx+1] = sin_a * q[2*idx]   + cos_a * q[2*idx+1]
        return q_rot

    # Yöntem 2: Vektörel form (PyTorch'ta kullanılan)
    def rope_vectorized(q, m, theta):
        d = len(q)
        # Pozisyon-frekans çarpımı
        m_theta = m * theta                          # (d/2,)
        cos_vals = np.cos(m_theta)                   # (d/2,)
        sin_vals = np.sin(m_theta)                   # (d/2,)

        # rotate_half: (q0,q1,q2,q3,...) → (-q1,q0,-q3,q2,...)
        q_even = q[0::2]   # (d/2,)
        q_odd  = q[1::2]   # (d/2,)

        q_rot = np.zeros(d)
        q_rot[0::2] = q_even * cos_vals - q_odd  * sin_vals
        q_rot[1::2] = q_even * sin_vals + q_odd  * cos_vals
        return q_rot

    q_rot1 = rope_matrix(q, m, theta)
    q_rot2 = rope_vectorized(q, m, theta)

    print(f"\nRoPE(q, m={m}):")
    print(f"  Matris yöntemi:     {np.round(q_rot1, 4)}")
    print(f"  Vektörel yöntem:    {np.round(q_rot2, 4)}")
    print(f"  Aynı mı? {np.allclose(q_rot1, q_rot2)}")
    print(f"  Norm korunuyor: ||q||={np.linalg.norm(q):.4f}, ||q_rot||={np.linalg.norm(q_rot1):.4f}")


# ─────────────────────────────────────────────────────────────
# 3. GÖRECELİLİK ÖZELLİĞİNİN KANITI
# ─────────────────────────────────────────────────────────────
# RoPE'nin en kritik özelliği:
#
# ⟨RoPE(q, m), RoPE(k, n)⟩ = g(q, k, m-n)
#
# Yani iç çarpım SADECE göreceli konum farkı (m-n)'ye bağlıdır.
# Mutlak pozisyon değil!
#
# Kanıt (2D için):
#   q' = R_m q,   k' = R_n k
#   q' · k' = (R_m q)^T (R_n k)
#           = q^T R_m^T R_n k
#           = q^T R_{n-m} k          (çünkü R^T R = R_{θ2-θ1})
#           = q^T R_{-(m-n)} k
#
# → Bu, m ve n'e değil sadece (m-n)'ye bağlıdır! ✓

def goreceligin_kaniti():
    print("\n" + "=" * 65)
    print("3. GÖRECELİLİK ÖZELLİĞİNİN KANITI")
    print("=" * 65)

    d, base = 16, 10000.0
    i = np.arange(d // 2)
    theta = base ** (-2 * i / d)

    def rope_vec(v, pos, theta):
        d = len(v)
        m_theta = pos * theta
        cos_v = np.cos(m_theta)
        sin_v = np.sin(m_theta)
        q_even = v[0::2]; q_odd = v[1::2]
        out = np.zeros(d)
        out[0::2] = q_even * cos_v - q_odd * sin_v
        out[1::2] = q_even * sin_v + q_odd * cos_v
        return out

    np.random.seed(42)
    q = np.random.randn(d)
    k = np.random.randn(d)

    # Farklı (m, n) çiftleri için ⟨RoPE(q,m), RoPE(k,n)⟩ hesapla
    # Eğer görecelilik geçerliyse: aynı (m-n)'ye sahip çiftler aynı skoru vermeli
    print("Görecelilik testi: ⟨RoPE(q,m), RoPE(k,n)⟩ sadece (m-n)'ye bağlı olmalı")
    print(f"\n{'(m, n)':>10}  {'m-n':>6}  {'İç Çarpım':>14}")
    print("-" * 35)

    test_pairs = [(0,0), (1,1), (3,3), (0,1), (1,2), (3,4), (0,3), (5,8)]
    for m, n in test_pairs:
        q_rot = rope_vec(q, m, theta)
        k_rot = rope_vec(k, n, theta)
        dot = np.dot(q_rot, k_rot)
        print(f"  ({m:2d},{n:2d})   {m-n:>6}  {dot:>14.6f}")

    print("\nGözlem: m-n=0 olan tüm çiftler aynı değere yakın →")
    print("        m-n=-1 olan tüm çiftler aynı değere yakın → ✓ Görecelilik")

    # Tam sayısal doğrulama
    # f(q, k, Δ) = q^T R_{-Δ} k  formülünü hesapla
    delta = 1
    pairs_same_delta = [(0,1), (5,6), (10,11)]
    dots = []
    for m, n in pairs_same_delta:
        assert m - n == -delta
        q_rot = rope_vec(q, m, theta)
        k_rot = rope_vec(k, n, theta)
        dots.append(np.dot(q_rot, k_rot))
    print(f"\nΔ={-delta} için iç çarpımlar: {[f'{d:.8f}' for d in dots]}")
    print(f"Hepsi aynı mı? {np.allclose(dots, dots[0])}")


# ─────────────────────────────────────────────────────────────
# 4. FREKANS ANALİZİ
# ─────────────────────────────────────────────────────────────
# θ_i = 10000^{-2i/d} = exp(-2i * ln(10000) / d)
#
# i=0 (küçük):     θ_0 = 1.0          → periyot = 2π token
# i=d/4 (orta):    θ_{d/4} = 0.01    → periyot = 628 token
# i=d/2-1 (büyük): θ_{d/2-1} ≈ 1e-4  → periyot = 62832 token
#
# Bu hiyerarşi:
#   Düşük boyutlar → kısa mesafe (local) ilişkileri kodlar
#   Yüksek boyutlar → uzun mesafe (global) ilişkileri kodlar

def frekans_analizi():
    print("\n" + "=" * 65)
    print("4. RoPE FREKANS ANALİZİ")
    print("=" * 65)

    configs = [
        ("LLaMA-2 7B",  d_k := 128, 10000.0),
        ("LLaMA-3 8B",  128,         500000.0),  # LLaMA-3'te base artırıldı!
        ("Mistral 7B",  128,         10000.0),
    ]

    for model_name, d_k, base in configs:
        print(f"\n{model_name}: d_k={d_k}, base={base:.0f}")
        i = np.arange(d_k // 2)
        theta = base ** (-2 * i / d_k)
        periods = 2 * np.pi / theta

        print(f"  {'Boyut i':>8}  {'θ_i':>12}  {'Periyot (token)':>18}")
        print(f"  " + "-" * 42)
        for idx in [0, d_k//8, d_k//4, d_k//2-2, d_k//2-1]:
            print(f"  {idx:>8}  {theta[idx]:>12.6f}  {periods[idx]:>18.1f}")

    # Base değişiminin etkisi
    print("\n--- BASE DEĞERİNİN ETKİSİ ---")
    d_k = 128
    print(f"{'Base':>12}  {'En kısa periyot':>18}  {'En uzun periyot':>18}")
    print("-" * 52)
    for base in [1000, 10000, 100000, 500000, 1000000]:
        theta = base ** (-2 * np.arange(d_k//2) / d_k)
        periods = 2 * np.pi / theta
        print(f"{base:>12,}  {periods[0]:>18.1f}  {periods[-1]:>18,.0f}")

    print("\nLLaMA-3'te base=500000 seçilmesi:")
    print("  → Maksimum periyot ~3M token → 128K context penceresi kolayca")
    print("  → LLaMA-2 (base=10000) → ~62K token periyot → 4K ctx'te sınırlı")


# ─────────────────────────────────────────────────────────────
# 5. SIFIRDAN TAM RoPE IMPLEMENTASYONu
# ─────────────────────────────────────────────────────────────

def rope_sifirdan():
    print("\n" + "=" * 65)
    print("5. SIFIRDAN TAM RoPE IMPLEMENTASYONU")
    print("=" * 65)

    class RotaryEmbedding:
        """
        RoPE — NumPy ile sıfırdan.
        Önbellekli cos/sin tablosu.
        """
        def __init__(self, d_k, max_seq=4096, base=10000.0):
            self.d_k = d_k
            self.base = base
            # θ_i = base^{-2i/d_k}
            i = np.arange(d_k // 2)
            self.theta = base ** (-2 * i / d_k)   # (d_k/2,)

            # Önbellekli cos/sin: (max_seq, d_k/2)
            positions = np.arange(max_seq)
            angles = np.outer(positions, self.theta)   # (max_seq, d_k/2)
            self.cos_cache = np.cos(angles)  # (max_seq, d_k/2)
            self.sin_cache = np.sin(angles)

        def rotate_half(self, x):
            """
            (x₀, x₁, x₂, x₃, ...) → (-x₁, x₀, -x₃, x₂, ...)
            """
            x_even = x[..., 0::2]   # (..., d/2)
            x_odd  = x[..., 1::2]   # (..., d/2)
            out = np.zeros_like(x)
            out[..., 0::2] = -x_odd
            out[..., 1::2] =  x_even
            return out

        def forward(self, x, offset=0):
            """
            x: (batch, n_heads, seq_len, d_k)
            offset: KV cache kullanırken başlangıç konumu
            """
            seq = x.shape[-2]
            cos = self.cos_cache[offset:offset+seq]   # (seq, d/2)
            sin = self.sin_cache[offset:offset+seq]   # (seq, d/2)

            # cos/sin → (d_k,) haline getir (çiftleri tekrar et)
            cos_full = np.repeat(cos, 2, axis=-1)   # (seq, d_k)
            sin_full = np.repeat(sin, 2, axis=-1)   # (seq, d_k)

            # Broadcast için boyut ekle: (1, 1, seq, d_k)
            cos_full = cos_full[np.newaxis, np.newaxis, :, :]
            sin_full = sin_full[np.newaxis, np.newaxis, :, :]

            return x * cos_full + self.rotate_half(x) * sin_full

    # Demo
    np.random.seed(42)
    d_k, batch, n_heads, seq = 64, 2, 4, 8
    rope = RotaryEmbedding(d_k, max_seq=512)

    q = np.random.randn(batch, n_heads, seq, d_k)
    k = np.random.randn(batch, n_heads, seq, d_k)

    q_rot = rope.forward(q)
    k_rot = rope.forward(k)

    print(f"Giriş: {q.shape}")
    print(f"Çıktı: {q_rot.shape}")
    print(f"Norm korunuyor: ||q[0,0,0]||={np.linalg.norm(q[0,0,0]):.4f}, "
          f"||q_rot[0,0,0]||={np.linalg.norm(q_rot[0,0,0]):.4f}")

    # Attention scores ile görecelilik
    # score(m, n) = q_rot[m] · k_rot[n]
    # Sadece (m-n)'ye bağlı olmalı
    print("\nAttention score görecelilik doğrulaması:")
    head = 0
    for delta in [0, 1, 2]:
        dots = []
        for start in range(4):
            m, n = start, start + delta
            if n < seq:
                d = np.dot(q_rot[0, head, m], k_rot[0, head, n])
                dots.append(d)
        variance = np.var(dots) if len(dots) > 1 else 0
        print(f"  Δ={delta}: dot products = {[f'{d:.4f}' for d in dots]},  "
              f"variance={variance:.6f}  ({'≈0 ✓' if variance < 0.5 else 'BEKLENENDEN BÜYÜK'})")


# ─────────────────────────────────────────────────────────────
# 6. PYTORCH RoPE IMPLEMENTASYONu
# ─────────────────────────────────────────────────────────────

def pytorch_rope():
    print("\n" + "=" * 65)
    print("6. PYTORCH RoPE IMPLEMENTASYONU")
    print("=" * 65)

    try:
        import torch

        class RotaryEmbedding(torch.nn.Module):
            def __init__(self, d_k, max_seq=4096, base=10000.0):
                super().__init__()
                i = torch.arange(d_k // 2).float()
                theta = 1.0 / (base ** (2 * i / d_k))
                self.register_buffer('theta', theta)

                # Önbellek cos/sin tablosu
                positions = torch.arange(max_seq).float()
                angles = torch.outer(positions, theta)   # (max_seq, d/2)
                self.register_buffer('cos_cache', angles.cos())
                self.register_buffer('sin_cache', angles.sin())

            @staticmethod
            def rotate_half(x):
                x1 = x[..., 0::2]
                x2 = x[..., 1::2]
                return torch.stack([-x2, x1], dim=-1).flatten(-2)

            def forward(self, x, offset=0):
                """x: (batch, n_heads, seq, d_k)"""
                seq = x.shape[2]
                cos = self.cos_cache[offset:offset+seq]  # (seq, d/2)
                sin = self.sin_cache[offset:offset+seq]

                # (seq, d_k) — çiftleri tekrar et
                cos = cos.repeat_interleave(2, dim=-1)[None, None]  # (1,1,seq,d)
                sin = sin.repeat_interleave(2, dim=-1)[None, None]

                return x * cos + self.rotate_half(x) * sin

        torch.manual_seed(0)
        d_k, batch, n_heads, seq = 64, 2, 4, 12
        rope = RotaryEmbedding(d_k, max_seq=512)

        q = torch.randn(batch, n_heads, seq, d_k)
        k = torch.randn(batch, n_heads, seq, d_k)

        q_rot = rope(q)
        k_rot = rope(k)

        print(f"Giriş: {q.shape}")
        print(f"Çıktı: {q_rot.shape}")
        print(f"Norm korunuyor: {torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)}")

        # Flash Attention ile kullanım
        out = torch.nn.functional.scaled_dot_product_attention(
            q_rot, k_rot, torch.randn(batch, n_heads, seq, d_k), is_causal=True
        )
        print(f"Flash Attention çıktı: {out.shape}")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    rotasyon_temeli()
    rope_formul()
    goreceligin_kaniti()
    frekans_analizi()
    rope_sifirdan()
    pytorch_rope()
