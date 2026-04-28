"""
=============================================================
MODÜL 6.3 — GRADIENT TRICKS VE EĞİTİM TEKNİKLERİ
=============================================================

LLM eğitiminde kullanılan kritik teknikler:
  1. Gradient Clipping
  2. Mixed Precision Training (FP16/BF16)
  3. Gradient Checkpointing (activation recomputation)
  4. Gradient Accumulation
  5. Flash Attention (verimli attention hesabı)
  6. Initialization strategies

Konular hem kavramsal hem PyTorch implementasyonu içerir.
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. GRADIENT CLIPPING
# ─────────────────────────────────────────────────────────────
# Problem: Exploding gradients → loss spike → eğitim çöküşü
#
# Çözüm: Global gradient norm'u kırp
#   gnorm = ||∇θ||₂ = √(Σ ||∂L/∂θ_i||²)
#   if gnorm > max_norm:
#       ∇θ ← ∇θ * (max_norm / gnorm)
#
# Bu, gradyanın yönünü korur, sadece büyüklüğünü kısıtlar.
# max_norm = 1.0 → LLM eğitiminde standart
#
# NOT: value clipping (her parametreyi ayrı kırp) kullanılmaz
#      çünkü gradyan yönünü bozar.

def gradient_clipping_demo():
    print("=" * 60)
    print("1. GRADIENT CLIPPING")
    print("=" * 60)

    def clip_gradients(gradients, max_norm=1.0):
        """
        gradients: parametre ismi → gradyan değeri sözlüğü
        Yerinde günceller ve orijinal normu döndürür.
        """
        # Global norm hesapla
        total_norm_sq = sum(np.sum(g**2) for g in gradients.values())
        total_norm = np.sqrt(total_norm_sq)

        # Kırp
        if total_norm > max_norm:
            scale = max_norm / total_norm
            for name in gradients:
                gradients[name] *= scale
            return total_norm, True
        return total_norm, False

    # Patlayan gradyan simülasyonu
    np.random.seed(42)
    scenarios = [
        ("Normal gradyan",    {"W1": np.random.randn(10) * 0.1, "W2": np.random.randn(10) * 0.1}),
        ("Büyük gradyan",     {"W1": np.random.randn(10) * 5.0,  "W2": np.random.randn(10) * 5.0}),
        ("Patlayan gradyan",  {"W1": np.random.randn(10) * 100., "W2": np.random.randn(10) * 100.}),
    ]

    max_norm = 1.0
    print(f"max_norm = {max_norm}")
    print(f"\n{'Senaryo':22s}  {'Orig Norm':>12}  {'Clip?':>8}  {'Final Norm':>12}")
    print("-" * 58)

    for name, grads in scenarios:
        orig_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        norm, clipped = clip_gradients(grads, max_norm)
        final_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        print(f"{name:22s}  {orig_norm:>12.4f}  {'Evet' if clipped else 'Hayır':>8}  {final_norm:>12.4f}")

    # Yön korunuyor mu?
    print("\nYön korunması doğrulaması:")
    g_orig = {"W": np.array([3.0, 4.0])}   # norm=5
    g_orig_copy = {"W": np.array([3.0, 4.0])}
    clip_gradients(g_orig, max_norm=1.0)
    unit_orig = g_orig_copy["W"] / np.linalg.norm(g_orig_copy["W"])
    unit_clip = g_orig["W"] / np.linalg.norm(g_orig["W"])
    print(f"  Orijinal yön: {unit_orig}")
    print(f"  Kırpılmış yön: {unit_clip}")
    print(f"  Aynı mı? {np.allclose(unit_orig, unit_clip)}")


# ─────────────────────────────────────────────────────────────
# 2. MIXED PRECISION — FP16 / BF16
# ─────────────────────────────────────────────────────────────
# FP32: 32-bit float, range 1.18e-38 ~ 3.4e38, 23-bit mantissa
# FP16: 16-bit float, range 6e-5 ~ 65504, 10-bit mantissa → overflow riski
# BF16: 16-bit, range = FP32 (8 exponent bit), 7-bit mantissa
#
# LLM eğitimi: BF16 (A100, H100) veya FP16 + loss scaling
# Neden BF16 tercih edilir?
#   - FP16: overflow riski (65504 limit), loss scaling gerektirir
#   - BF16: FP32 ile aynı range → daha stabil
#
# Mixed Precision stratejisi:
#   - Forward + Backward: BF16/FP16 (hız için)
#   - Optimizer state (m, v): FP32 (hassasiyet için)
#   - Weight master copy: FP32
#
# Bellek tasarrufu:
#   - FP32 parametre: 4 byte × N
#   - BF16 parametre: 2 byte × N  → %50 tasarruf
#   - Optimizer state: BF16 + FP32 = 2+4+4+4 = 14 byte/param
#                      FP32 only   = 4+4+4+4 = 16 byte/param

def mixed_precision_aciklama():
    print("\n" + "=" * 60)
    print("2. MIXED PRECISION EĞİTİM")
    print("=" * 60)

    print("""
  Sayı formatları:
    FP32:  (-1)^s × 2^(e-127) × (1 + mantissa/2^23)
           range: 1.18e-38 ~ 3.40e38
    FP16:  range: 6.1e-5 ~ 65504  (çok küçük!)
    BF16:  range: 1.18e-38 ~ 3.40e38  (FP32 ile aynı)
           ama 7-bit mantissa → daha az hassasiyet
    """)

    # Overflow örneği
    import struct

    def float_to_bits(f, precision='fp32'):
        if precision == 'fp32':
            bits = struct.pack('f', f)
            return int.from_bytes(bits, 'little')
        return None

    max_fp16 = 65504.0
    print(f"FP16 max değer: {max_fp16}")
    print(f"FP16 overflow: {max_fp16 * 1.01}  → NaN/Inf!")
    print(f"BF16 max değer: ~3.4e38 (FP32 ile aynı)")

    # Bellek analizi
    print("\n7B parametre model bellek analizi:")
    n = 7_000_000_000

    fp32_params = n * 4
    bf16_params = n * 2
    adam_state  = n * 4 * 2  # m ve v (FP32)
    grad_bf16   = n * 2

    # Mixed precision (BF16 fwd/bwd + FP32 optim state)
    mixed = bf16_params + fp32_params + adam_state + grad_bf16
    # Full FP32
    full_fp32 = fp32_params + adam_state + n*4  # fp32 grads

    print(f"  Full FP32:       {full_fp32/1e9:.1f} GB")
    print(f"  Mixed Precision: {mixed/1e9:.1f} GB  ({mixed/full_fp32*100:.0f}%)")
    print(f"  → Tasarruf: {(full_fp32-mixed)/1e9:.1f} GB")


def pytorch_mixed_precision():
    print("\n--- PYTORCH AMP (Automatic Mixed Precision) ---")

    try:
        import torch
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512))
        optimizer = torch.optim.AdamW(model.parameters())

        # BF16 (A100+)
        if torch.cuda.is_available():
            model = model.cuda()
            x = torch.randn(32, 512).cuda()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(x)
            print(f"BF16 output dtype: {out.dtype}")
        else:
            print("CUDA yok — CPU üzerinde FP32 kullanılacak.")
            x = torch.randn(32, 512)
            out = model(x)
            print(f"FP32 output dtype: {out.dtype}")

        print("PyTorch AMP kullanımı:")
        print("""
  # GPU ile tipik kullanım:
  scaler = torch.cuda.amp.GradScaler()  # FP16 için
  with torch.autocast(device_type='cuda', dtype=torch.float16):
      loss = model(x)
  scaler.scale(loss).backward()
  scaler.unscale_(optimizer)
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  scaler.step(optimizer)
  scaler.update()
        """)

    except ImportError:
        print("PyTorch yüklü değil.")


# ─────────────────────────────────────────────────────────────
# 3. GRADIENT CHECKPOINTING (Activation Recomputation)
# ─────────────────────────────────────────────────────────────
# Problem: Backprop için forward'da hesaplanan tüm activations saklanır
#   → Bellek = O(n_layers × seq_len × d_model)
#
# Çözüm: Sadece seçili "checkpoint" aktivasyonları sakla
#         Geri kalanı backward'da yeniden hesapla
#
# Tradeoff: %33 daha fazla hesap ↔ %65 bellek tasarrufu

def gradient_checkpointing_aciklama():
    print("\n" + "=" * 60)
    print("3. GRADIENT CHECKPOINTING")
    print("=" * 60)

    print("""
  Normal Backprop:
    Forward:  A₁ → A₂ → ... → A_L → Loss
              (tüm aktivasyonlar bellekte tutulur)
    Backward: Loss → A_L → ... → A₁
              (kaydedilen aktivasyonlar kullanılır)
    Bellek: O(L)

  Checkpoint Backprop:
    Forward:  A₁ → [A₂] → A₃ → [A₄] → ... → Loss
              (köşeli parantezler: kaydedilen checkpointler)
    Backward: Kaydedilmemiş aktivasyonları yeniden hesapla
    Bellek: O(√L)  [optimal checkpointing]
    Hesap: %33 fazla (bazı forward pass'ler tekrar edilir)

  LLaMA-3 70B eğitiminde:
    - Checkpoint olmadan: ~280 GB activation memory
    - Checkpoint ile: ~90 GB activation memory
    → 4×H100 yerine 1×H100 ile eğitilebilir
    """)

    # Bellek analizi
    n_layers = 32
    seq_len  = 2048
    d_model  = 4096
    batch    = 4
    n_heads  = 32
    d_ff     = 4 * d_model

    bytes_per_element = 2  # BF16

    # Per-layer activation memory
    attn_qkv = batch * seq_len * d_model * 3   # Q, K, V
    attn_w   = batch * n_heads * seq_len * seq_len  # attention weights
    ffn_act  = batch * seq_len * d_ff         # FFN activation
    residual = batch * seq_len * d_model      # residual stream

    per_layer = (attn_qkv + attn_w + ffn_act + residual) * bytes_per_element

    total_normal = n_layers * per_layer
    total_ckpt   = int(np.sqrt(n_layers)) * per_layer  # √L checkpoints

    print(f"\nLLaMA-7B benzeri bellek analizi:")
    print(f"  L={n_layers}, d={d_model}, seq={seq_len}, batch={batch}")
    print(f"  Katman başı aktivasyon: {per_layer/1e6:.1f} MB")
    print(f"  Normal (L katman): {total_normal/1e9:.2f} GB")
    print(f"  Checkpoint (√L ≈{int(np.sqrt(n_layers))} katman): {total_ckpt/1e9:.2f} GB")
    print(f"  Tasarruf: {(total_normal-total_ckpt)/1e9:.2f} GB (%{(1-total_ckpt/total_normal)*100:.0f})")


# ─────────────────────────────────────────────────────────────
# 4. FLASH ATTENTION — VERIMLI ATTENTION
# ─────────────────────────────────────────────────────────────
# Dao et al. (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention"
#
# Normal attention:
#   scores = Q @ K^T   → materialize (n,n) matrix → O(n²) memory
#   weights = softmax(scores)
#   output = weights @ V
#
# FlashAttention: tile-based hesap
#   - Dikkat matrisini hiç materialize etme
#   - Blok blok işle → O(n) memory
#   - HBM (High Bandwidth Memory) yerine SRAM'de hesapla → hız
#
# FlashAttention-2 (Dao 2023): 2x daha hızlı
# FlashAttention-3 (Shah 2024): H100 için optimize

def flash_attention_aciklama():
    print("\n" + "=" * 60)
    print("4. FLASH ATTENTION")
    print("=" * 60)

    print("""
  Standart Attention:
    1. S = Q @ K^T         (n×n matris — GPU'da yavaş okuma/yazma)
    2. P = softmax(S)
    3. O = P @ V
    Bellek: O(n²),  IO: O(n²)

  FlashAttention:
    - S'yi hiç bellekte tutma
    - Online softmax ile blok blok hesapla
    - Tüm işlemi SRAM'de yap (HBM'e gidip gelmeden)
    Bellek: O(n),   IO: O(n² / M × M) = O(n²) ama pratik 5-20x hızlı

  Neden HBM sinyalı önemli?
    GPU SRAM: ~192 KB, 19 TB/s bandwidth
    GPU HBM:  ~80 GB,  2 TB/s bandwidth
    → SRAM 10x daha hızlı → oraya sığdır!
    """)

    # Bellek karşılaştırması
    for seq_len in [512, 1024, 2048, 4096, 8192]:
        n_heads, d_k = 32, 128
        batch = 4

        attn_normal = batch * n_heads * seq_len * seq_len * 2  # bytes (BF16)
        attn_flash  = batch * n_heads * seq_len * d_k * 2      # sadece Q veya K

        print(f"  seq={seq_len:5d}: Normal={attn_normal/1e6:.0f}MB, Flash={attn_flash/1e6:.0f}MB  ({attn_normal/attn_flash:.0f}x tasarruf)")


# ─────────────────────────────────────────────────────────────
# 5. WEIGHT INITIALIZATION
# ─────────────────────────────────────────────────────────────
# Başlangıç ağırlıkları önemli!
# Çok büyük → exploding activations
# Çok küçük → vanishing activations
#
# GPT-2 initialization:
#   Genel: N(0, 0.02)
#   Residual projection: N(0, 0.02/√(2L))  — L katman sayısı
#   Neden /√(2L)? Her katman residual ekliyor, varyansı kontrol et

def initialization_demo():
    print("\n" + "=" * 60)
    print("5. WEIGHT INITIALIZATION")
    print("=" * 60)

    n_layers = 12
    d_model  = 768

    # GPT-2 init: N(0, 0.02)
    # Residual projection: N(0, 0.02/√(2L))
    std_general   = 0.02
    std_residual  = 0.02 / np.sqrt(2 * n_layers)

    print(f"GPT-2 initialization (L={n_layers}, d={d_model}):")
    print(f"  Genel std:             {std_general:.5f}")
    print(f"  Residual proj std:     {std_residual:.5f}  (= 0.02/√{2*n_layers})")

    # Xavier / Glorot
    def xavier_std(fan_in, fan_out):
        return np.sqrt(2.0 / (fan_in + fan_out))

    # He (Kaiming)
    def he_std(fan_in):
        return np.sqrt(2.0 / fan_in)

    print(f"\nLinear(d={d_model}, d_ff={4*d_model}):")
    print(f"  Xavier std: {xavier_std(d_model, 4*d_model):.5f}")
    print(f"  He std:     {he_std(d_model):.5f}")
    print(f"  GPT-2 std:  {std_general:.5f}")

    # Aktivasyon istatistiklerini kontrol et
    np.random.seed(0)
    L = 12
    x = np.random.randn(1, 10, d_model)

    print(f"\n{L} katmanlı ağda aktivasyon büyüklüğü:")
    print(f"  (Residual olmadan, sadece matris çarpımı)")
    for std in [0.02, 0.1, std_residual]:
        act = x.copy()
        for _ in range(L):
            W = np.random.randn(d_model, d_model) * std
            act = act @ W
        print(f"  std={std:.5f}: activation norm = {np.linalg.norm(act):.4f}")


if __name__ == "__main__":
    gradient_clipping_demo()
    mixed_precision_aciklama()
    pytorch_mixed_precision()
    gradient_checkpointing_aciklama()
    flash_attention_aciklama()
    initialization_demo()
