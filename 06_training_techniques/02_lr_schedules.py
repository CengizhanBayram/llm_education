"""
=============================================================
MODÜL 6.2 — ÖĞRENME HIZI ÇİZELGELERİ (LR Schedules)
=============================================================

LLM eğitiminde LR schedule kritiktir:
  - Çok büyük LR → loss spike, instabil
  - Çok küçük LR → yavaş öğrenme
  - Warmup + decay → en iyi pratik

Kullanılan schedule'lar:
  - Constant: sabit LR (artık kullanılmıyor)
  - Linear decay: GPT-1
  - Cosine decay + warmup: GPT-3, LLaMA, Mistral — STANDART
  - WSD (Warmup-Stable-Decay): MiniCPM, yeni trend
  - Cyclical: araştırma ortamında

Konular:
  1. Warmup neden gerekli?
  2. Cosine decay
  3. WSD schedule
  4. Öğrenme hızı bulmak (LR finder)
  5. PyTorch scheduler'lar
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. WARMUP NEDEN GEREKLİ?
# ─────────────────────────────────────────────────────────────
# Başlangıçta:
#   - Parametre rastgele → gradyanlar kararlı değil
#   - Adam'ın 2. moment v_t sıfırdan başlıyor → unreliable update direction
#   - Büyük LR → büyük ama yanlış yönde adımlar → instabil
#
# Warmup: LR'yi yavaşça artır
#   - t=0: LR=0  →  t=T_warmup: LR=peak
#   - Adam v_t birikme fırsatı buluyor
#   - Gradient istikrarı kazanılıyor
#
# Standart: warmup = %1-2 toplam adım
# GPT-3: 375M warmup tokens / 300B total = %0.1

def warmup_aciklama():
    print("=" * 60)
    print("1. WARMUP NEDENİ")
    print("=" * 60)

    T_warmup = 100
    T_total  = 1000
    lr_max   = 3e-4
    lr_min   = lr_max * 0.1

    print(f"Toplam adım: {T_total}, Warmup: {T_warmup}, LR_max: {lr_max}")

    # Adam bias düzeltmesi ile warmup bağlantısı
    beta2 = 0.95
    effective_lr_factor = []
    for t in range(1, 21):
        bias_correction = np.sqrt(1 - beta2**t)  # √(1 - β2^t)
        effective_lr_factor.append(bias_correction)

    print("\nAdam β2=0.95: t adımda v̂ bias düzeltmesi √(1-0.95^t):")
    for t in [1, 5, 10, 20]:
        print(f"  t={t:2d}: {effective_lr_factor[t-1]:.4f}  "
              f"({'kararlı değil' if effective_lr_factor[t-1] < 0.5 else 'kararlı'})")


# ─────────────────────────────────────────────────────────────
# 2. LR SCHEDULE FONKSİYONLARI
# ─────────────────────────────────────────────────────────────

def get_cosine_lr(step, T_warmup, T_total, lr_max, lr_min=None):
    """
    Warmup + Cosine Decay schedule (GPT-3, LLaMA)
    step ∈ [0, T_total]
    """
    lr_min = lr_min or lr_max * 0.1

    if step < T_warmup:
        # Lineer warmup
        return lr_max * step / T_warmup

    if step >= T_total:
        return lr_min

    # Cosine decay
    progress = (step - T_warmup) / (T_total - T_warmup)
    cosine_coeff = 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_min + cosine_coeff * (lr_max - lr_min)


def get_wsd_lr(step, T_warmup, T_stable_end, T_total, lr_max, lr_min=None):
    """
    WSD: Warmup → Stable → Cosine Decay (MiniCPM)
    """
    lr_min = lr_min or lr_max * 0.1

    if step < T_warmup:
        return lr_max * step / T_warmup

    if step < T_stable_end:
        return lr_max  # sabit

    if step >= T_total:
        return lr_min

    progress = (step - T_stable_end) / (T_total - T_stable_end)
    cosine_coeff = 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_min + cosine_coeff * (lr_max - lr_min)


def schedule_gorsellestir():
    print("\n" + "=" * 60)
    print("2. LR SCHEDULE KARŞILAŞTIRMASI")
    print("=" * 60)

    T_total  = 1000
    T_warmup = 100
    lr_max   = 3e-4
    lr_min   = 3e-5

    # Cosine
    steps = np.arange(T_total)
    cosine_lrs = [get_cosine_lr(s, T_warmup, T_total, lr_max, lr_min) for s in steps]

    # WSD
    T_stable = 700
    wsd_lrs = [get_wsd_lr(s, T_warmup, T_stable, T_total, lr_max, lr_min) for s in steps]

    # Constant (no schedule)
    const_lrs = [lr_max] * T_total

    print(f"{'Adım':>6}  {'Cosine':>10}  {'WSD':>10}  {'Constant':>10}")
    print("-" * 42)
    for s in [0, 50, 100, 300, 500, 700, 800, 900, 999]:
        print(f"{s:>6}  {cosine_lrs[s]:>10.2e}  {wsd_lrs[s]:>10.2e}  {const_lrs[s]:>10.2e}")

    # Karşılaştırma yorumu
    print(f"\n{'Schedule':12s}  {'Warmup':>8}  {'Davranış':>30}")
    print("-" * 55)
    info = [
        ("Cosine",    "Var",  "Yavaş, sürekli decay"),
        ("WSD",       "Var",  "Sabit + Hızlı son decay"),
        ("Constant",  "Yok", "Decay yok — genellikle kötü"),
        ("Linear",    "Var",  "Lineer decay (eski GPT)"),
    ]
    for name, wu, desc in info:
        print(f"{name:12s}  {wu:>8}  {desc}")


# ─────────────────────────────────────────────────────────────
# 3. WSD SCHEDULE AVANTAJI
# ─────────────────────────────────────────────────────────────

def wsd_avantaji():
    print("\n" + "=" * 60)
    print("3. WSD SCHEDULE AVANTAJI (MiniCPM 2024)")
    print("=" * 60)

    print("""
  Cosine schedule sorunu:
    - LR sürekli düşüyor → her checkpoint farklı LR
    - Eğitimi ortadan "devam ettirmek" zor
    - LR ne zaman düşürülmeli? Önceden karar vermelisin

  WSD (Warmup-Stable-Decay) çözümü:
    - Stable aşama: düşük LR olmadan uzun süre eğit
    - Decay aşama: son %10 adımda hızla düşür
    - Herhangi bir checkpoint'ten devam edilebilir
    - Veri karışımı (data mixture) değiştirmek için esnek

  MiniCPM bulgularından:
    - WSD, Cosine ile aynı final loss'a ulaşıyor
    - AMA: eğitim sırasında "restart" noktaları oluşturulabiliyor
    - Sürekli eğitim (continual pretraining) için ideal

  Kullanım önerisi (2024):
    - Short runs: Cosine yeterli
    - Long runs / continual: WSD tercih et
    """)


# ─────────────────────────────────────────────────────────────
# 4. LR FINDER (Smith 2015)
# ─────────────────────────────────────────────────────────────
# LR'yi deneysel bul:
#   1. LR'yi 1e-7'den başlat
#   2. Her batch'te LR'yi küçük çarp ile artır
#   3. Loss vs LR'yi izle
#   4. Loss en hızlı düştüğü LR → iyi LR
#   5. Loss patlamadan önce → max LR

def lr_finder_demo():
    print("\n" + "=" * 60)
    print("4. LR FINDER")
    print("=" * 60)

    np.random.seed(42)

    # Basit parabol loss fonksiyonu simülasyonu
    # f(θ) = (θ - 1)² + 0.1 * noise
    # Optimal LR: ~0.1-0.5

    theta = np.array([3.0])
    lr_start = 1e-5
    lr_end   = 10.0
    n_iters  = 100
    beta = 0.98   # loss smoothing için

    lr_values = np.geomspace(lr_start, lr_end, n_iters)
    losses = []
    smoothed_loss = 0.0

    for i, lr in enumerate(lr_values):
        # Simüle edilmiş loss ve gradyan
        g = 2 * (theta - 1.0) + 0.5 * np.random.randn(1)
        loss = (theta[0] - 1.0)**2

        # Loss smoothing
        smoothed_loss = beta * smoothed_loss + (1 - beta) * loss
        losses.append(smoothed_loss / (1 - beta**(i+1)))

        # Güncelleme
        theta = theta - lr * g

        # Patlama kontrolü
        if smoothed_loss > 10 * losses[0] + 1:
            break

    min_idx = np.argmin(losses)
    print(f"En düşük loss LR'si: {lr_values[min_idx]:.2e}")
    print(f"Önerilen LR: {lr_values[min_idx] / 10:.2e}  (min'in 1/10'u)")

    print("\nLR → Loss tablosu (seçilmiş noktalar):")
    print(f"{'LR':>10}  {'Loss':>10}")
    selected = [0, 20, 40, 50, 60, 70, min_idx, min(min_idx+10, len(losses)-1)]
    for i in sorted(set(selected)):
        if i < len(losses):
            print(f"{lr_values[i]:>10.2e}  {losses[i]:>10.6f}")


# ─────────────────────────────────────────────────────────────
# 5. PYTORCH SCHEDULER'LAR
# ─────────────────────────────────────────────────────────────

def pytorch_schedulers():
    print("\n" + "=" * 60)
    print("5. PYTORCH SCHEDULER'LAR")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        model = nn.Linear(10, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Warmup + Cosine — SequentialLR ile
        T_warmup = 100
        T_total  = 1000

        scheduler_warmup  = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                       total_iters=T_warmup)
        scheduler_cosine  = CosineAnnealingLR(optimizer, T_max=T_total - T_warmup,
                                               eta_min=3e-5)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine],
                                  milestones=[T_warmup])

        print(f"Warmup + Cosine LR schedule:")
        for step in [0, 50, 100, 300, 500, 700, 999]:
            for _ in range(1 if step == 0 else step - (prev_step if step > 0 else 0)):
                scheduler.step()
            prev_step = step
            lr = optimizer.param_groups[0]['lr']
            print(f"  Adım {step:4d}: LR = {lr:.2e}")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    warmup_aciklama()
    schedule_gorsellestir()
    wsd_avantaji()
    lr_finder_demo()
    pytorch_schedulers()
