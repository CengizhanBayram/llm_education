"""
=============================================================
MODÜL 7.2 — BERT MİMARİSİ
=============================================================

BERT (Devlin et al., 2019) — "Bidirectional Encoder Representations from Transformers"

GPT'den farkı:
  - GPT: decoder-only, left-to-right (causal)
  - BERT: encoder-only, bidirectional (tüm tokenlar herkesi görür)

Eğitim görevleri:
  1. MLM (Masked Language Modeling): token'ların %15'ini maskele, tahmin et
  2. NSP (Next Sentence Prediction): iki cümle ard arda mı?

Sonraki nesil geliştirmeler:
  - RoBERTa: NSP kaldırıldı, daha uzun eğitim
  - DeBERTa: Disentangled attention (konum ve içerik ayrı)
  - ALBERT: Parametre paylaşımı ile küçültülmüş BERT
=============================================================
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False


def bert_vs_gpt():
    print("=" * 65)
    print("BERT vs GPT — TEMEL FARK")
    print("=" * 65)

    print("""
  GPT (Causal / Autoregressive):
    Token 0: Yalnızca token 0'ı görür
    Token 1: Token 0 ve 1'i görür
    Token 2: Token 0, 1, 2'yi görür
    ...
    Attention mask: Alt üçgen (causal mask)

    Kullanım: Metin üretimi (generation)

  BERT (Bidirectional):
    Token 0: TÜM tokenlara bakabilir
    Token 1: TÜM tokenlara bakabilir
    ...
    Attention mask: Tüm konumlar (padding hariç)

    Kullanım: Metin anlama (classification, NER, QA)

  Analoji:
    BERT → Bir metni OKUR ve anlar (iki yönlü bağlam)
    GPT  → Bir metni YAZAR (sadece geçmişe bakarak)
    """)


# ─────────────────────────────────────────────────────────────
# MLM — MASKED LANGUAGE MODELING
# ─────────────────────────────────────────────────────────────
# Eğitim prosedürü:
#   1. Token'ların %15'ini seç
#   2. Seçilenlerin:
#      - %80'ini [MASK] token ile değiştir
#      - %10'unu rastgele başka token ile değiştir
#      - %10'unu olduğu gibi bırak (model bilmemeli ki maskelendi)
#   3. Model orijinal tokeni tahmin eder
#
# Neden %80/%10/%10?
#   - Sadece [MASK] kullansaydık: model fine-tune'da [MASK] görmez → mismatch
#   - Rastgele değiştirme: modeli dikkatli olmaya zorlar
#   - Olduğu gibi bırakma: modelin her tokeni temsil etmesini zorlar

def mlm_demo():
    print("\n" + "=" * 65)
    print("MLM — MASKED LANGUAGE MODELING")
    print("=" * 65)

    np.random.seed(42)

    tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
    print(f"Orijinal: {tokens}")

    # %15 maske
    mask_prob = 0.15
    n_tokens  = len(tokens)
    selected  = np.where(np.random.rand(n_tokens) < mask_prob)[0]

    # Eğer hiç seçilmediyse en az bir tane seç (demo için)
    if len(selected) == 0:
        selected = np.array([2])

    masked_tokens = tokens.copy()
    labels = {}

    for idx in selected:
        r = np.random.rand()
        labels[idx] = tokens[idx]  # hedef
        if r < 0.8:
            masked_tokens[idx] = "[MASK]"
        elif r < 0.9:
            masked_tokens[idx] = np.random.choice(tokens)
        # else: olduğu gibi bırak

    print(f"Maskelenmiş: {masked_tokens}")
    print(f"Tahmin edilecek: {labels}")
    print(f"\nMLM loss: sadece maskelenen tokenlar için CE hesaplanır")
    print(f"→ {len(selected)}/{n_tokens} token = %{len(selected)/n_tokens*100:.0f} bu adımda eğitildi")

    # Verimlilik notu
    print(f"\nMLM vs CLM verimlilik:")
    print(f"  BERT MLM:  %15 token/batch öğrenilir")
    print(f"  GPT CLM:   %100 token/batch öğrenilir (her token tahmin edilir)")
    print(f"  → GPT eğitim sinyali ~6.7x daha yoğun!")
    print(f"  → BERT bunu bidirectionality ile telafi eder")


# ─────────────────────────────────────────────────────────────
# NSP — NEXT SENTENCE PREDICTION
# ─────────────────────────────────────────────────────────────
# İki cümle alır: [CLS] A [SEP] B [SEP]
# %50: B, A'nın gerçek devamı (IsNext)
# %50: B, rastgele cümle (NotNext)
# [CLS] tokenının çıktısı ikili sınıflandırma için kullanılır
#
# NOT: RoBERTa (2019) NSP'nin zararlı olduğunu gösterdi!
#      Kaldırılınca performans arttı.

def nsp_aciklama():
    print("\n" + "=" * 65)
    print("NSP — NEXT SENTENCE PREDICTION")
    print("=" * 65)

    print("""
  Format:
    [CLS] A [SEP] B [SEP]
    ↓
    [CLS] çıktısı → Linear → sigmoid → IsNext/NotNext

  Örnek:
    IsNext:   [CLS] The cat sat [SEP] It was on a mat [SEP]
    NotNext:  [CLS] The cat sat [SEP] The weather is nice [SEP]

  Problemler (RoBERTa 2019):
    - NSP çok kolay → model gerçekten öğrenmiyor
    - Topic prediction'a dönüşüyor (aynı konu mu? → IsNext)
    - Kaldırılınca BERT benchmark'ları iyileşti

  Sonuç: Modern BERT varyantları (RoBERTa, DeBERTa) NSP kullanmaz.
    """)


# ─────────────────────────────────────────────────────────────
# BERT INPUT REPRESENTATION
# ─────────────────────────────────────────────────────────────
# BERT girişi 3 embedding'in toplamı:
#   1. Token Embedding:    WordPiece token'ı
#   2. Segment Embedding:  Cümle A (0) veya Cümle B (1)
#   3. Position Embedding: Öğrenilmiş konum

def bert_input_representation():
    print("\n" + "=" * 65)
    print("BERT GİRİŞ TEMSİLİ")
    print("=" * 65)

    print("""
  Input = TokenEmb + SegmentEmb + PositionEmb

  Örnek: "[CLS] hello world [SEP] how are you [SEP]"
  Token IDs:    [101, 7592, 2088,  102, 2129, 2024, 2017,  102]
  Segment IDs:  [0,   0,    0,     0,   1,    1,    1,     1  ]
  Position IDs: [0,   1,    2,     3,   4,    5,    6,     7  ]

  [CLS] = 101: Classification token
         → Son katmanda bu tokenın çıktısı classification için kullanılır
         → Tüm sekansın "özeti"

  [SEP] = 102: Separator token
         → İki cümleyi ayırır

  Special tokens BERT'e özgü, GPT'de yok!
    """)

    if TORCH:
        vocab_size = 30522   # BERT WordPiece vocab
        d_model    = 768
        max_len    = 512
        n_segments = 2

        tok_emb = nn.Embedding(vocab_size, d_model)
        seg_emb = nn.Embedding(n_segments, d_model)
        pos_emb = nn.Embedding(max_len, d_model)
        ln      = nn.LayerNorm(d_model)

        emb_params = tok_emb.weight.numel() + seg_emb.weight.numel() + pos_emb.weight.numel()
        print(f"BERT embedding parametreleri:")
        print(f"  Token:    {tok_emb.weight.shape}  → {tok_emb.weight.numel():,}")
        print(f"  Segment:  {seg_emb.weight.shape}  → {seg_emb.weight.numel():,}")
        print(f"  Position: {pos_emb.weight.shape} → {pos_emb.weight.numel():,}")
        print(f"  Toplam:   {emb_params:,}")


# ─────────────────────────────────────────────────────────────
# ROBERTA ve DEBERTA FARKLILIKLARI
# ─────────────────────────────────────────────────────────────

def bert_varyantlari():
    print("\n" + "=" * 65)
    print("BERT VARYANTLARı")
    print("=" * 65)

    print("""
  ┌───────────────┬──────────────────────────────────────────────────────┐
  │ Model         │ Temel Fark                                           │
  ├───────────────┼──────────────────────────────────────────────────────┤
  │ BERT          │ Orijinal. MLM + NSP.                                │
  │               │ 110M (base) / 340M (large)                          │
  ├───────────────┼──────────────────────────────────────────────────────┤
  │ RoBERTa       │ NSP kaldırıldı.                                     │
  │               │ Daha uzun eğitim (160GB veri vs 16GB).              │
  │               │ Dynamic masking (her epoch yeni mask).              │
  │               │ +5-8 puan GLUE'da                                   │
  ├───────────────┼──────────────────────────────────────────────────────┤
  │ ALBERT        │ Cross-layer parameter sharing (tüm katmanlar aynı W)│
  │               │ Factorized embedding (E ayrı, küçük)                │
  │               │ BERT'ten 18x daha az parametre, yakın performans    │
  ├───────────────┼──────────────────────────────────────────────────────┤
  │ DeBERTa       │ Disentangled Attention:                             │
  │               │   Content vektörü ve Position vektörü ayrı          │
  │               │   Attention: c2c + c2p + p2c (3 bileşen)           │
  │               │ 2021: SuperGLUE'de insanı geçti                     │
  ├───────────────┼──────────────────────────────────────────────────────┤
  │ ModernBERT    │ 2024. RoPE + Flash Attention + Global tokens.       │
  │               │ 8192 context length.                                │
  │               │ Encoder mimarisini yeniden canlandırdı.             │
  └───────────────┴──────────────────────────────────────────────────────┘
    """)


def pytorch_bert_blok():
    if not TORCH:
        return

    print("\n" + "=" * 65)
    print("PYTORCH BERT ENCODER BLOĞU")
    print("=" * 65)

    class BERTEncoderBlock(nn.Module):
        """Bidirectional: mask yok → tüm tokenlar birbirini görür."""
        def __init__(self, d_model=768, n_heads=12, d_ff=3072, dropout=0.1):
            super().__init__()
            # BERT Post-LN kullanır (orijinal)
            self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
            self.ln1   = nn.LayerNorm(d_model)
            self.ffn   = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            self.ln2   = nn.LayerNorm(d_model)

        def forward(self, x, key_padding_mask=None):
            # Post-LN Self-Attention (causal mask YOK)
            h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
            x = self.ln1(x + h)
            # Post-LN FFN
            x = self.ln2(x + self.ffn(x))
            return x

    torch.manual_seed(0)
    blok = BERTEncoderBlock()
    params = sum(p.numel() for p in blok.parameters())
    x = torch.randn(2, 15, 768)
    out = blok(x)
    print(f"BERT Encoder Bloğu: {params:,} parametre")
    print(f"  Giriş: {x.shape}  →  Çıktı: {out.shape}")
    print(f"  Causal mask YOK → tüm tokenlar herkesi görüyor")


if __name__ == "__main__":
    bert_vs_gpt()
    mlm_demo()
    nsp_aciklama()
    bert_input_representation()
    bert_varyantlari()
    pytorch_bert_blok()
