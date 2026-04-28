"""
=============================================================
MODÜL 5.3 — DİL MODELİ EĞİTİM DÖNGÜSÜ
=============================================================

Gerçek bir LLM eğitim döngüsünün tüm bileşenleri:
  1. Data pipeline (tokenizasyon, batching, shuffle)
  2. Forward pass + Loss
  3. Backward pass + Gradient clipping
  4. Optimizer adımı
  5. Değerlendirme (perplexity, loss curve)
  6. Checkpoint kayıt/yükleme

Bu modülde ayrıca:
  - Gradient accumulation (büyük batch simülasyonu)
  - Learning rate warmup
  - Mixed precision (kavramsal)
=============================================================
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def egitim_dongusu_kavrami():
    print("=" * 60)
    print("DİL MODELİ EĞİTİM DÖNGÜSÜ KAVRAMI")
    print("=" * 60)

    print("""
  Veri: Büyük metin corpus (internet, kitaplar, kod...)
        Tokenize edilip ID dizisine çevrilir.

  Batch oluşturma:
    Uzun token dizisi → sabit uzunluklu parçalar
    x = tokens[i : i+block_size]      (giriş)
    y = tokens[i+1 : i+block_size+1]  (hedef — 1 kaydırılmış)

  Loss: Cross-Entropy
    Her token için: -log P(doğru_token | bağlam)
    Ortalama: batch içindeki tüm tokenlara göre

  Eğitim adımı:
    1. forward(x) → logits
    2. loss = CE(logits, y)
    3. loss.backward() → gradyanlar
    4. clip_grad_norm() → patlayan gradyanları kes
    5. optimizer.step() → parametreleri güncelle
    6. optimizer.zero_grad() → gradyanları sıfırla

  Metriklere bak:
    - train loss: düşüyor mu?
    - val loss: train loss ile fark büyüyor mu? (overfitting)
    - perplexity = exp(loss): yorumlaması kolay
    """)


def numpy_egitim_ornegi():
    print("\n" + "=" * 60)
    print("NUMPY EĞİTİM DÖNGÜSÜ (Basitleştirilmiş)")
    print("=" * 60)

    # Çok basit dil modeli: tek gizli katman
    # P(w_t | w_{t-1}) — bigram modeli, sinir ağı ile
    np.random.seed(42)

    vocab_size = 10
    d_model    = 16

    # Parametreler
    W_emb  = np.random.randn(vocab_size, d_model) * 0.01  # embedding
    W_out  = np.random.randn(d_model, vocab_size) * 0.01  # output

    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def forward(x_ids):
        # x_ids: (batch,) integer
        emb = W_emb[x_ids]          # (batch, d)
        logits = emb @ W_out        # (batch, vocab)
        probs = softmax(logits)
        return logits, probs

    def ce_loss(probs, y_ids):
        n = len(y_ids)
        return -np.mean(np.log(probs[np.arange(n), y_ids] + 1e-12))

    # Sentetik veri: sıralı token çiftleri
    tokens = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 5)  # tekrar eden desen

    lr = 0.1
    batch = 8
    losses = []

    for step in range(200):
        # Random batch
        idx = np.random.randint(0, len(tokens)-1, size=batch)
        x = tokens[idx]
        y = tokens[idx + 1]

        # Forward
        logits, probs = forward(x)
        loss = ce_loss(probs, y)

        # Backward (gradyanlar)
        # dL/dlogits = (probs - one_hot(y)) / batch
        dlogits = probs.copy()
        dlogits[np.arange(batch), y] -= 1
        dlogits /= batch

        # dL/dW_out = emb^T @ dlogits
        emb = W_emb[x]
        dW_out = emb.T @ dlogits

        # dL/dW_emb (sadece x'e karşılık gelen satırlar)
        d_emb = dlogits @ W_out.T
        dW_emb = np.zeros_like(W_emb)
        np.add.at(dW_emb, x, d_emb)

        # Güncelleme
        W_out  -= lr * dW_out
        W_emb  -= lr * dW_emb

        losses.append(loss)
        if (step + 1) % 50 == 0:
            ppl = np.exp(loss)
            print(f"  Adım {step+1:4d}: loss={loss:.4f}, PPL={ppl:.2f}")

    print(f"\nBaşlangıç PPL: {np.exp(losses[0]):.2f}")
    print(f"Son PPL:       {np.exp(losses[-1]):.2f}")
    print(f"→ Desen öğrenildi: {vocab_size} sınıf için min PPL = 1.0")


def pytorch_tam_egitim():
    if not TORCH_AVAILABLE:
        return

    print("\n" + "=" * 60)
    print("PYTORCH TAM EĞİTİM DÖNGÜSÜ")
    print("=" * 60)

    torch.manual_seed(42)

    # ─────────────────────────────────────────────────────────
    # Küçük GPT (önceki modülden)
    # ─────────────────────────────────────────────────────────
    vocab_size = 65
    block_size = 32
    batch_size = 32
    d_model    = 64
    n_heads    = 4
    n_layers   = 2
    d_ff       = 256
    dropout    = 0.1

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1  = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
            self.ln2  = nn.LayerNorm(d_model)
            self.ffn  = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout))

        def forward(self, x):
            T = x.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(T)
            h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
            x = x + h
            x = x + self.ffn(self.ln2(x))
            return x

    class GPTMini(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(block_size, d_model)
            self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
            self.ln_f   = nn.LayerNorm(d_model)
            self.head   = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, x, y=None):
            B, T = x.shape
            h = self.tok(x) + self.pos(torch.arange(T))
            h = self.ln_f(self.blocks(h))
            logits = self.head(h)
            loss = None
            if y is not None:
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            return logits, loss

    model = GPTMini()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parametre")

    # ─────────────────────────────────────────────────────────
    # Sentetik veri (örüntü öğrenme)
    # ─────────────────────────────────────────────────────────
    # Basit tekrarlayan desen: [0,1,2,...,64, 0,1,2,...]
    num_tokens = 5000
    synthetic_data = torch.arange(num_tokens) % vocab_size

    def get_batch():
        idx = torch.randint(num_tokens - block_size, (batch_size,))
        x = torch.stack([synthetic_data[i:i+block_size] for i in idx])
        y = torch.stack([synthetic_data[i+1:i+block_size+1] for i in idx])
        return x, y

    # ─────────────────────────────────────────────────────────
    # Optimizer + LR Schedule (Warmup + Cosine)
    # ─────────────────────────────────────────────────────────
    max_iters    = 300
    warmup_iters = 30
    lr_max  = 3e-3
    lr_min  = lr_max * 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max,
                                   betas=(0.9, 0.95), weight_decay=0.1)

    def get_lr(it):
        # Warmup: lineer artış
        if it < warmup_iters:
            return lr_max * it / warmup_iters
        # Cosine decay
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + np.cos(np.pi * progress))
        return lr_min + coeff * (lr_max - lr_min)

    # ─────────────────────────────────────────────────────────
    # Gradient Accumulation
    # ─────────────────────────────────────────────────────────
    # Gerçek batch=128 ama bellek sadece batch=32 alıyor
    # Çözüm: 4 adım boyunca gradyanları biriktir, sonra güncelle
    accum_steps = 4
    effective_batch = batch_size * accum_steps
    print(f"\nGradient Accumulation: {batch_size} × {accum_steps} = {effective_batch} efektif batch")

    # ─────────────────────────────────────────────────────────
    # Eğitim döngüsü
    # ─────────────────────────────────────────────────────────
    print(f"\n{'Iter':>6}  {'LR':>10}  {'Loss':>10}  {'PPL':>8}")
    print("-" * 40)

    model.train()
    optimizer.zero_grad()

    for iter_i in range(1, max_iters + 1):
        # LR güncelle
        lr_now = get_lr(iter_i)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # Gradient accumulation
        total_loss = 0
        for micro_step in range(accum_steps):
            x, y = get_batch()
            _, loss = model(x, y)
            loss = loss / accum_steps  # ölçekle
            loss.backward()
            total_loss += loss.item()

        # Gradient clipping — patlayan gradyanları önle
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        if iter_i % 60 == 0 or iter_i == 1:
            ppl = np.exp(total_loss)
            print(f"{iter_i:>6}  {lr_now:>10.2e}  {total_loss:>10.4f}  {ppl:>8.2f}  "
                  f"(gnorm={grad_norm:.3f})")

    # ─────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────
    print("\n--- CHECKPOINT ---")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': max_iters,
        'config': {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_layers': n_layers,
        }
    }
    import os, tempfile
    ckpt_path = os.path.join(tempfile.gettempdir(), 'gpt_mini.pt')
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint kaydedildi: {ckpt_path}")

    # Yükle
    loaded = torch.load(ckpt_path, map_location='cpu')
    print(f"Checkpoint yüklendi: iter={loaded['iter']}, config={loaded['config']}")


def egitim_ipuclari():
    print("\n" + "=" * 60)
    print("EĞİTİM İPUÇLARI — LLM RESEARCHER İÇİN")
    print("=" * 60)

    print("""
  1. LOSS İZLEME
     - Train loss her adımda düşmeli
     - Val loss > train loss → aşırı öğrenme
     - Loss spike → LR çok büyük veya kötü batch
     - Loss plateau → LR çok küçük veya model kapasitesi yetersiz

  2. ÖĞRENME HIZI
     - Cosine schedule + warmup: modern standarttır
     - Warmup: ~%1-2 toplam adım
     - Peak LR: 3e-4 ile 3e-3 arası (model büyüklüğüne bağlı)
     - Chinchilla: max LR ≈ 0.3 / √(d_model)

  3. GRADIENT CLIPPING
     - max_norm = 1.0: standart değer
     - Clip'lenen norm sürekli büyükse → LR düşür veya batch büyüt

  4. BATCH SIZE
     - Büyük batch → kararlı gradyan, ama daha az güncellem
     - Gradient accumulation ile küçük GPU'da büyük batch
     - LLaMA-2 70B: batch=4M token

  5. SCALING LAWS (Chinchilla 2022)
     - N model parametresi, D eğitim tokeni
     - Optimal: D ≈ 20 × N  (örn. 7B model → 140B token)
     - GPT-3 (175B): 300B token eğitildi → fazla eğitilmemiş
     - LLaMA-2 (7B): 2T token → Chinchilla'dan 14x fazla eğitim
    """)


if __name__ == "__main__":
    egitim_dongusu_kavrami()
    numpy_egitim_ornegi()
    pytorch_tam_egitim()
    egitim_ipuclari()
