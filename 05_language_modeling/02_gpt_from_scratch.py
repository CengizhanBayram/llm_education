"""
=============================================================
MODÜL 5.2 — GPT SIFIRDAN (Karakter Seviyesi)
=============================================================

Karpathy'nin "makemore" ve "nanoGPT" çalışmasından ilham alınmıştır.
Gerçek bir GPT modeli, karakter seviyesinde, tam PyTorch ile.

Hedef: "Shakespeare" gibi metin üretebilen küçük bir model.

Mimari:
  - Decoder-only Transformer (GPT tarzı)
  - Karakter seviyesi tokenizasyon (26+boşluk+noktalama ≈ 65 token)
  - Pre-LN, GELU, weight tying
  - ~100K-500K parametre (laptop'ta çalışır)
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
    print("PyTorch yüklü değil. 'pip install torch' ile yükleyin.")


def run_gpt_demo():
    if not TORCH_AVAILABLE:
        print("PyTorch gerekli.")
        return

    torch.manual_seed(1337)

    # ─────────────────────────────────────────────────────────
    # Küçük eğitim verisi — sabit metin
    # ─────────────────────────────────────────────────────────
    text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;"""

    # Tokenizasyon (karakter seviyesi)
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: ''.join(itos[i] for i in ids)

    print("=" * 60)
    print("KARAKTER SEVİYESİ GPT")
    print("=" * 60)
    print(f"Vocab: {vocab_size} karakter: {''.join(chars[:20])}...")
    print(f"Veri uzunluğu: {len(text)} karakter")

    data = torch.tensor(encode(text), dtype=torch.long)

    # Train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]

    # ─────────────────────────────────────────────────────────
    # Hiperparametreler (küçük model)
    # ─────────────────────────────────────────────────────────
    block_size = 64    # bağlam penceresi
    batch_size = 16
    d_model    = 64
    n_heads    = 4
    n_layers   = 3
    d_ff       = 256
    dropout    = 0.1
    lr         = 3e-3
    max_iters  = 500

    # ─────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────
    class Head(nn.Module):
        def __init__(self, head_size):
            super().__init__()
            self.Wq = nn.Linear(d_model, head_size, bias=False)
            self.Wk = nn.Linear(d_model, head_size, bias=False)
            self.Wv = nn.Linear(d_model, head_size, bias=False)
            self.drop = nn.Dropout(dropout)
            self.register_buffer('mask',
                torch.tril(torch.ones(block_size, block_size)))

        def forward(self, x):
            B, T, C = x.shape
            q = self.Wq(x)
            k = self.Wk(x)
            v = self.Wv(x)
            w = q @ k.transpose(-2, -1) / (C ** 0.5)
            w = w.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
            w = F.softmax(w, dim=-1)
            w = self.drop(w)
            return w @ v

    class MultiHeadAttn(nn.Module):
        def __init__(self, n_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
            self.proj  = nn.Linear(d_model, d_model)
            self.drop  = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            return self.drop(self.proj(out))

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1  = nn.LayerNorm(d_model)
            self.attn = MultiHeadAttn(n_heads, d_model // n_heads)
            self.ln2  = nn.LayerNorm(d_model)
            self.ffn  = FeedForward()

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x

    class GPTChar(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size, d_model)
            self.blocks  = nn.Sequential(*[Block() for _ in range(n_layers)])
            self.ln_f    = nn.LayerNorm(d_model)
            self.head    = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok = self.tok_emb(idx)
            pos = self.pos_emb(torch.arange(T))
            x = tok + pos
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.head(x)   # (B, T, vocab)

            if targets is None:
                return logits, None

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )
            return logits, loss

        def generate(self, idx, max_new=100, temperature=1.0):
            for _ in range(max_new):
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
            return idx

    model = GPTChar()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parametre")
    print(f"  d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, d_ff={d_ff}")

    # ─────────────────────────────────────────────────────────
    # Eğitim
    # ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def get_batch(split):
        data_split = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_split) - block_size, (batch_size,))
        x = torch.stack([data_split[i:i+block_size] for i in ix])
        y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
        return x, y

    @torch.no_grad()
    def estimate_loss(eval_iters=50):
        model.eval()
        losses = {}
        for split in ['train', 'val']:
            loss_list = []
            for _ in range(eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                loss_list.append(loss.item())
            losses[split] = np.mean(loss_list)
        model.train()
        return losses

    print(f"\nEğitim başlıyor ({max_iters} iterasyon)...")
    print(f"{'İter':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'PPL':>8}")
    print("-" * 44)

    for iter_i in range(max_iters):
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iter_i % 100 == 0 or iter_i == max_iters - 1:
            losses = estimate_loss()
            ppl = np.exp(losses['val'])
            print(f"{iter_i:>6}  {losses['train']:>12.4f}  {losses['val']:>12.4f}  {ppl:>8.2f}")

    # ─────────────────────────────────────────────────────────
    # Üretim
    # ─────────────────────────────────────────────────────────
    print("\n--- ÜRETIM ---")
    model.eval()
    start = torch.tensor([[stoi.get('T', 0)]], dtype=torch.long)

    for temp in [0.5, 1.0, 1.5]:
        with torch.no_grad():
            generated = model.generate(start.clone(), max_new=150, temperature=temp)
        text_out = decode(generated[0].tolist())
        print(f"\nTemperature={temp}:")
        print(f"  '{text_out[:100]}...'")


if __name__ == "__main__":
    run_gpt_demo()
