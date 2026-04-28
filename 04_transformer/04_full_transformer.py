"""
=============================================================
MODÜL 4.4 — TAM TRANSFORMER MİMARİSİ
=============================================================

"Attention Is All You Need" (Vaswani et al., 2017) — tam implementasyon
ve GPT tarzı decoder-only model.

Bu modül şunları içerir:
  1. Orijinal Encoder-Decoder Transformer
  2. GPT tarzı Decoder-Only model
  3. Mimariler arası karşılaştırma
  4. PyTorch ile GPT-2 small
=============================================================
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
# Yardımcı bileşenler (tüm modülden toplu import yerine inline)
# ──────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def layer_norm(x, gamma, beta, eps=1e-5):
    mu  = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def mha_forward(Q_in, K_in, V_in, Wq, Wk, Wv, Wo, n_heads, mask=None):
    B, Tq, d = Q_in.shape
    Tk = K_in.shape[1]
    dk = d // n_heads

    q = (Q_in @ Wq).reshape(B, Tq, n_heads, dk).transpose(0, 2, 1, 3)
    k = (K_in @ Wk).reshape(B, Tk, n_heads, dk).transpose(0, 2, 1, 3)
    v = (V_in @ Wv).reshape(B, Tk, n_heads, dk).transpose(0, 2, 1, 3)

    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk)
    if mask is not None:
        scores = np.where(mask[None, None], scores, -1e9)
    w = softmax(scores)
    ctx = (w @ v).transpose(0, 2, 1, 3).reshape(B, Tq, d)
    return ctx @ Wo

def ffn_forward(x, W1, b1, W2, b2):
    return gelu(x @ W1 + b1) @ W2 + b2


# ─────────────────────────────────────────────────────────────
# 1. GPT TARZI DECODER-ONLY MODEL
# ─────────────────────────────────────────────────────────────
# Mimari:
#   Token Embedding    E ∈ R^{V x d}
#   + Position Embedding P ∈ R^{T_max x d}
#   ↓
#   [GPT Block] × L     (masked self-attn + FFN, her ikisi Pre-LN ile)
#   ↓
#   Final Layer Norm
#   ↓
#   LM Head: Linear(d, V)  — çoğu zaman E ile ağırlıklar paylaşılır

class GPT:
    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff=None):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        d_ff = d_ff or 4 * d_model

        s = 0.02  # GPT-2 initialization: N(0, 0.02)

        # Embeddings
        self.token_emb = np.random.randn(vocab_size, d_model) * s
        self.pos_emb   = np.random.randn(max_len, d_model) * s

        # Katman parametreleri (her L için)
        self.layers = []
        for i in range(n_layers):
            W_scale = s / np.sqrt(2 * n_layers)  # residual projection ölçeklemesi
            layer = {
                # Layer Norm 1
                "ln1_g": np.ones(d_model),
                "ln1_b": np.zeros(d_model),
                # Self-Attention
                "Wq": np.random.randn(d_model, d_model) * s,
                "Wk": np.random.randn(d_model, d_model) * s,
                "Wv": np.random.randn(d_model, d_model) * s,
                "Wo": np.random.randn(d_model, d_model) * W_scale,
                # Layer Norm 2
                "ln2_g": np.ones(d_model),
                "ln2_b": np.zeros(d_model),
                # FFN
                "W1": np.random.randn(d_model, d_ff) * s,
                "b1": np.zeros(d_ff),
                "W2": np.random.randn(d_ff, d_model) * W_scale,
                "b2": np.zeros(d_model),
            }
            self.layers.append(layer)

        # Final Layer Norm
        self.ln_f_g = np.ones(d_model)
        self.ln_f_b = np.zeros(d_model)

        # LM Head — token_emb ile ağırlık paylaşımı (weight tying)
        # Projeksiyonu ayrı tutmak yerine token_emb^T kullan

    def forward(self, input_ids):
        """
        input_ids: (batch, seq) — token indices
        Returns: logits (batch, seq, vocab_size)
        """
        batch, seq = input_ids.shape
        assert seq <= self.max_len

        # 1. Embedding
        tok = self.token_emb[input_ids]   # (batch, seq, d)
        pos = self.pos_emb[:seq]           # (seq, d) → broadcast
        x = tok + pos                      # (batch, seq, d)

        # 2. Causal mask
        causal = np.tril(np.ones((seq, seq), dtype=bool))

        # 3. Transformer blokları
        for layer in self.layers:
            # Pre-LN Self-Attention
            x_norm = layer_norm(x, layer["ln1_g"], layer["ln1_b"])
            attn = mha_forward(
                x_norm, x_norm, x_norm,
                layer["Wq"], layer["Wk"], layer["Wv"], layer["Wo"],
                self.n_heads, mask=causal
            )
            x = x + attn

            # Pre-LN FFN
            x_norm = layer_norm(x, layer["ln2_g"], layer["ln2_b"])
            x = x + ffn_forward(x_norm, layer["W1"], layer["b1"],
                                         layer["W2"], layer["b2"])

        # 4. Final LN
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)

        # 5. LM Head (weight tying: E^T)
        logits = x @ self.token_emb.T   # (batch, seq, vocab)
        return logits

    def param_count(self):
        emb_p = self.vocab_size * self.d_model + self.max_len * self.d_model
        d, d_ff = self.d_model, 4 * self.d_model
        per_layer = (4 * d * d +           # Wq, Wk, Wv, Wo
                     2 * d * d_ff + d_ff + d +  # FFN
                     2 * 2 * d)            # 2 LayerNorm
        return emb_p + self.n_layers * per_layer + 2 * self.d_model


def gpt_demo():
    print("=" * 60)
    print("TAM GPT TARZI MODEL DEMO")
    print("=" * 60)

    np.random.seed(42)

    # Mini GPT (eğitime uygun boyutta)
    gpt = GPT(
        vocab_size=1000,
        max_len=64,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
    )
    print(f"Mini GPT: ~{gpt.param_count()/1e6:.2f}M parametre")

    input_ids = np.array([[5, 12, 43, 7, 9, 21, 100, 55]])
    logits = gpt.forward(input_ids)
    print(f"\nGiriş: {input_ids.shape}  (batch=1, seq=8)")
    print(f"Çıktı logit: {logits.shape}  (batch=1, seq=8, vocab=1000)")

    # İlk token tahminleri
    probs = softmax(logits[0, -1])  # son token olasılıkları
    top5 = np.argsort(probs)[::-1][:5]
    print(f"\nSon token için top-5 tahmin:")
    for t in top5:
        print(f"  token {t:4d}: p = {probs[t]:.6f}")


# ─────────────────────────────────────────────────────────────
# 2. GPT-2 SMALL PARAMETRESİ
# ─────────────────────────────────────────────────────────────

def gpt2_parametre():
    print("\n" + "=" * 60)
    print("GPT-2 SMALL PARAMETRE ANALİZİ")
    print("=" * 60)

    # GPT-2 small: 12 katman, d=768, 12 kafa, d_ff=3072
    vocab = 50257
    max_len = 1024
    L = 12
    d = 768
    n_heads = 12
    d_ff = 3072

    emb_token = vocab * d
    emb_pos   = max_len * d
    per_layer_attn = 4 * d * d   # Wq, Wk, Wv, Wo (bias dahil ~4d daha)
    per_layer_ffn  = 2 * d * d_ff + d_ff + d
    per_layer_ln   = 2 * 2 * d
    per_layer = per_layer_attn + per_layer_ffn + per_layer_ln
    ln_final = 2 * d
    lm_head  = vocab * d   # weight tying — aslında sıfır ek parametre

    total = emb_token + emb_pos + L * per_layer + ln_final

    print(f"{'Bileşen':30s}  {'Parametre':>12}")
    print("-" * 46)
    print(f"{'Token Embedding':30s}  {emb_token:>12,}")
    print(f"{'Position Embedding':30s}  {emb_pos:>12,}")
    print(f"{'Attention (her katman)':30s}  {per_layer_attn:>12,}")
    print(f"{'FFN (her katman)':30s}  {per_layer_ffn:>12,}")
    print(f"{'LayerNorm (her katman)':30s}  {per_layer_ln:>12,}")
    print(f"{'Final LayerNorm':30s}  {ln_final:>12,}")
    print("-" * 46)
    print(f"{'TOPLAM':30s}  {total:>12,}  (~{total/1e6:.0f}M)")
    print(f"\nNot: LM Head = Token Embedding^T → weight tying (ek parametre yok)")


# ─────────────────────────────────────────────────────────────
# 3. MİMARİ KARŞILAŞTIRMA
# ─────────────────────────────────────────────────────────────

def mimari_karsilastirma():
    print("\n" + "=" * 60)
    print("3. TRANSFORMER MİMARİ TÜRLERİ KARŞILAŞTIRMASI")
    print("=" * 60)

    print("""
  ┌──────────────────┬────────────────────────────────────────────────────┐
  │ Mimari           │ Açıklama                                           │
  ├──────────────────┼────────────────────────────────────────────────────┤
  │ Encoder-Decoder  │ Tam transformer. Çeviri, özetleme (T5, BART)       │
  │ (Seq2Seq)        │ Encoder: çift yönlü attention                      │
  │                  │ Decoder: masked self + cross-attention              │
  ├──────────────────┼────────────────────────────────────────────────────┤
  │ Encoder-Only     │ Metin sınıflandırma, NER, embedding (BERT, RoBERTa)│
  │                  │ Çift yönlü attention — gelecek token görülebilir   │
  │                  │ MLM (Masked Language Modeling) ile eğitim          │
  ├──────────────────┼────────────────────────────────────────────────────┤
  │ Decoder-Only     │ Metin üretimi, genel amaçlı LLM (GPT, LLaMA)      │
  │                  │ Sadece geçmiş tokenlar görülebilir (causal)        │
  │                  │ Next Token Prediction ile eğitim                   │
  │                  │ ← GÜNÜMÜZDE EN YAYGIN LLM MİMARİSİ               │
  └──────────────────┴────────────────────────────────────────────────────┘

  Günümüz büyük LLM'lerinin tamamı decoder-only:
    GPT-2/3/4, LLaMA 1/2/3, Mistral, Falcon, Gemma, Qwen, Phi, ...

  Neden decoder-only?
    1. Eğitim hedefi basit: sadece next token prediction
    2. Eğitim verisi: internet metni (etiket gerekmez)
    3. Few-shot prompting ile hızlı adaptasyon
    4. Ölçeklenince çok güçlü (scaling laws)
    """)


# ─────────────────────────────────────────────────────────────
# 4. PYTORCH İLE GPT-2 SMALL
# ─────────────────────────────────────────────────────────────

def pytorch_gpt():
    print("\n" + "=" * 60)
    print("4. PYTORCH İLE GPT-2 SMALL BENZERİ")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        torch.manual_seed(42)

        class GPTConfig:
            vocab_size = 50257
            max_len    = 1024
            d_model    = 768
            n_heads    = 12
            n_layers   = 12
            d_ff       = 3072
            dropout    = 0.1

        class GPTBlock(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.ln1  = nn.LayerNorm(cfg.d_model)
                self.attn = nn.MultiheadAttention(
                    cfg.d_model, cfg.n_heads,
                    dropout=cfg.dropout, batch_first=True)
                self.ln2  = nn.LayerNorm(cfg.d_model)
                self.ffn  = nn.Sequential(
                    nn.Linear(cfg.d_model, cfg.d_ff),
                    nn.GELU(),
                    nn.Linear(cfg.d_ff, cfg.d_model),
                    nn.Dropout(cfg.dropout),
                )

            def forward(self, x):
                T = x.shape[1]
                mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
                h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
                x = x + h
                x = x + self.ffn(self.ln2(x))
                return x

        class GPTModel(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
                self.pos_emb = nn.Embedding(cfg.max_len,    cfg.d_model)
                self.drop    = nn.Dropout(cfg.dropout)
                self.blocks  = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layers)])
                self.ln_f    = nn.LayerNorm(cfg.d_model)
                self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
                # Weight tying
                self.lm_head.weight = self.tok_emb.weight

            def forward(self, input_ids):
                B, T = input_ids.shape
                tok = self.tok_emb(input_ids)
                pos = self.pos_emb(torch.arange(T, device=input_ids.device))
                x = self.drop(tok + pos)
                for block in self.blocks:
                    x = block(x)
                x = self.ln_f(x)
                return self.lm_head(x)

        cfg = GPTConfig()
        model = GPTModel(cfg)
        total_params = sum(p.numel() for p in model.parameters())
        # Weight tying: lm_head.weight = tok_emb.weight (aynı parametre)
        unique_params = sum(p.numel() for p in set(model.parameters()))

        print(f"GPT-2 Small PyTorch:")
        print(f"  Toplam (paylaşımlı): {total_params:,}")
        print(f"  Benzersiz: {unique_params:,}  (~{unique_params/1e6:.0f}M)")

        # Küçük test
        ids = torch.randint(0, cfg.vocab_size, (1, 10))
        with torch.no_grad():
            logits = model(ids)
        print(f"\nTest: input={ids.shape} → logits={logits.shape}")

        # CE loss hesabı
        input_ids = torch.randint(0, cfg.vocab_size, (2, 20))
        targets = input_ids[:, 1:].contiguous()   # shift by 1
        logits = model(input_ids[:, :-1])
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
        print(f"Örnek CE loss: {loss.item():.4f}")
        print(f"Perplexity: {torch.exp(loss).item():.2f}  (rastgele model için ~{cfg.vocab_size:.0f})")

    except ImportError:
        print("PyTorch yüklü değil.")


if __name__ == "__main__":
    gpt_demo()
    gpt2_parametre()
    mimari_karsilastirma()
    pytorch_gpt()
