"""
=============================================================
MODÜL 5.1 — TOKENİZASYON VE BPE (Byte Pair Encoding)
=============================================================

LLM'ler ham metin yerine token'larla çalışır.
Tokenizasyon: metin → token ID dizisi

Yöntemler:
  - Karakter bazlı: her harf bir token (vocab küçük, seq uzun)
  - Kelime bazlı: her kelime bir token (vocab büyük, OOV sorunu)
  - Alt kelime (Sub-word): BPE, WordPiece, Unigram (uzlaşı noktası!)

GPT-2/3/4, LLaMA: BPE (Byte-level BPE)
BERT: WordPiece
T5, Llama (sentencepiece): Unigram

Konular:
  1. Neden tokenizasyon?
  2. BPE algoritması sıfırdan
  3. Byte-level BPE
  4. Tokenizasyon örnekleri
=============================================================
"""

from collections import Counter, defaultdict
import re

# ─────────────────────────────────────────────────────────────
# 1. NEDEN TOKENİZASYON?
# ─────────────────────────────────────────────────────────────

def neden_tokenizasyon():
    print("=" * 60)
    print("1. NEDEN TOKENİZASYON?")
    print("=" * 60)

    text = "I love transformers!"

    # Karakter bazlı
    chars = list(text)
    print(f"Karakter bazlı ({len(chars)} token): {chars}")

    # Kelime bazlı (basit)
    words = text.split()
    print(f"Kelime bazlı ({len(words)} token): {words}")

    # OOV (Out-of-Vocabulary) sorunu
    print(f"\nKelime bazlı OOV sorunu:")
    vocab = {"I", "love", "transformers"}
    test_word = "transforming"
    print(f"  Vocab'da 'transformers' var ama '{test_word}' yok → OOV!")

    # Alt kelime çözümü
    # "transforming" → "transform" + "ing" → her ikisi vocab'da
    subwords = ["transform", "##ing"]
    print(f"  Alt kelime çözümü: '{test_word}' → {subwords}")

    print(f"\nGPT-4 vocab: ~100,000 token")
    print(f"  'transformers' → ['transform', 'ers'] (2 token)")
    print(f"  'Constantinople' → ['Const', 'antine', 'ople'] (3 token)")
    print(f"  ' Hello' → [' Hello'] (1 token, başında boşluk!)")


# ─────────────────────────────────────────────────────────────
# 2. BPE ALGORİTMASI — SIFIRDAN
# ─────────────────────────────────────────────────────────────
# Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units"
#
# Algoritma:
#   1. Başlangıç vocab: karakter seti + </w> (kelime sonu işareti)
#   2. Kelime frekanslarını hesapla
#   3. En sık birleşen çifti bul
#   4. Bu çifti yeni token olarak ekle
#   5. Tüm kelime temsillerinde bu çifti birleştir
#   6. İstenen vocab büyüklüğüne ulaşana kadar 3-5'i tekrarla
#
# Örnek:
#   "low" → ["l", "o", "w", "</w>"]
#   "lower" → ["l", "o", "w", "e", "r", "</w>"]
#   Adım 1: ("l", "o") → "lo"  (en sık çift)
#   Adım 2: ("lo", "w") → "low"
#   ...

def get_vocab(corpus):
    """Kelime frekanslarını hesapla, kelimeleri karakter dizisine böl."""
    vocab = Counter()
    for word in corpus.split():
        # Her karakteri ayır + </w> ekle
        chars = list(word) + ['</w>']
        vocab[tuple(chars)] += 1
    return vocab

def get_stats(vocab):
    """Ardışık çiftlerin frekansını hesapla."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = list(word)
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """En iyi çifti tüm kelimelerle birleştir."""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in vocab.items():
        # Tuple → string, çifti değiştir, string → tuple
        word_str = ' '.join(word)
        new_word_str = word_str.replace(bigram, replacement)
        new_vocab[tuple(new_word_str.split())] = freq
    return new_vocab

def bpe_train(corpus, num_merges):
    """BPE eğitimi — birleştirme kurallarını öğren."""
    vocab = get_vocab(corpus)

    print(f"Başlangıç vocab (ilk 5):")
    for word, freq in list(vocab.items())[:5]:
        print(f"  {word}: {freq}")

    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break

        # En sık çifti seç
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)

        vocab = merge_vocab(best_pair, vocab)

        if i < 10 or i == num_merges - 1:
            print(f"  Birleştirme {i+1}: {best_pair} → {''.join(best_pair)}  (freq={pairs[best_pair]})")

    return merges, vocab


def bpe_encode(text, merges):
    """Öğrenilen BPE kurallarıyla metni tokenize et."""
    # Kelimelere böl ve karakter dizisine dönüştür
    words = [list(w) + ['</w>'] for w in text.split()]

    for pair in merges:
        bigram = ''.join(pair)
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        words = new_words

    # Tokenları düzleştir
    tokens = []
    for word in words:
        tokens.extend(word)
    return tokens


def bpe_demo():
    print("\n" + "=" * 60)
    print("2. BPE DEMO")
    print("=" * 60)

    corpus = """low lower lowest newest widest longest
                low low low lower lower lower lowest
                new new new newest newest widest"""

    print(f"Corpus: {corpus[:80]}...")
    print()

    merges, final_vocab = bpe_train(corpus, num_merges=15)

    print(f"\nFinal vocab (ilk 10):")
    for word, freq in sorted(final_vocab.items(), key=lambda x: -x[1])[:10]:
        print(f"  {''.join(word)}: {freq}")

    # Tokenizasyon
    test_words = ["low", "lower", "newest", "lowest", "highestwidest"]
    print(f"\nTokenizasyon örnekleri:")
    for word in test_words:
        tokens = bpe_encode(word, merges)
        print(f"  '{word}' → {tokens}")


# ─────────────────────────────────────────────────────────────
# 3. BYTE-LEVEL BPE (GPT-2, GPT-3, LLaMA)
# ─────────────────────────────────────────────────────────────
# GPT-2'de Radford et al. byte-level BPE kullanır:
#   - Karakterler yerine byte'lar (0-255) kullanılır
#   - Unicode desteği: her dil ve emoji tokenize edilebilir
#   - OOV yok: her byte sequence geçerli
#   - </w> yerine Ġ (önüne boşluk) kullanılır
#
# GPT-2 vocab: 50,257 token (50,000 BPE + 256 byte tokens + 1 <|endoftext|>)

def byte_level_bpe_ozet():
    print("\n" + "=" * 60)
    print("3. BYTE-LEVEL BPE (GPT-2/LLaMA)")
    print("=" * 60)

    print("""
  Byte-level BPE özellikleri:
    - Başlangıç vocab: 256 byte değeri (0x00 - 0xFF)
    - Türkçe, Çince, Arapça vs. sorunsuz çalışır
    - Boşluklar Ġ ile temsil edilir: "Hello world" → ["Hello", "Ġworld"]
    - GPT-2: 50,257 token
    - LLaMA: 32,000 token (SentencePiece BPE)
    - LLaMA-3: 128,256 token (tiktoken BPE)

  Token sayısı rehberi (yaklaşık):
    - 1 İngilizce kelime ≈ 1.3 token
    - 100 token ≈ 75 İngilizce kelime
    - Türkçe biraz daha fazla token
    """)

    # OpenAI tiktoken ile pratik örnek
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")   # GPT-4 tokenizer

        texts = [
            "Hello, world!",
            "Transformers are amazing.",
            "Merhaba dünya!",
            "Yapay zeka çok güçlü.",
            "1234567890",
        ]
        print("tiktoken (cl100k_base) tokenizasyonu:")
        for text in texts:
            tokens = enc.encode(text)
            decoded = [enc.decode([t]) for t in tokens]
            print(f"  '{text}'")
            print(f"    Tokens ({len(tokens)}): {tokens}")
            print(f"    Decoded: {decoded}")
    except ImportError:
        print("\ntiktoken yüklü değil. Örnek çıktı:")
        print("  'Hello, world!' → [15496, 11, 995, 0]  (4 token)")
        print("  'Transformers are amazing.' → [8291, 364, 389, 4998, 13]  (5 token)")


# ─────────────────────────────────────────────────────────────
# 4. SPECİAL TOKENS
# ─────────────────────────────────────────────────────────────
# LLM'ler özel tokenlar kullanır:
#   - <|endoftext|>: GPT-2'de metin sonu
#   - <s>, </s>: LLaMA'da başlangıç/bitiş
#   - <pad>: padding token
#   - [CLS], [SEP]: BERT'te özel görev tokenleri
#   - <|system|>, <|user|>, <|assistant|>: Chat LLM'lerde rol tokenleri

def special_tokens():
    print("\n" + "=" * 60)
    print("4. ÖZEL TOKENLAR")
    print("=" * 60)

    print("""
  GPT-2: <|endoftext|> (token id: 50256)
  LLaMA-2: <s>=1, </s>=2, <unk>=0
  LLaMA-3: <|begin_of_text|>, <|end_of_text|>, <|eot_id|>

  Chat formatı (LLaMA-3):
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a helpful assistant.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    What is 2+2?
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    The answer is 4.
    <|eot_id|>

  Model bu formatı eğitimde görür → inference'ta aynı formata uyar.
    """)


# ─────────────────────────────────────────────────────────────
# 5. TOKENİZASYON SORUNLARI
# ─────────────────────────────────────────────────────────────

def tokenizasyon_sorunlari():
    print("\n" + "=" * 60)
    print("5. TOKENİZASYON SORUNLARI")
    print("=" * 60)

    print("""
  Problem 1: Sayılar
    "9.11" < "9.9"? LLM karıştırıyor!
    Çünkü: "9.11" → ["9", ".", "1", "1"]  (4 token)
            "9.9"  → ["9", ".", "9"]       (3 token)
    Model token bazında işler, sayısal değeri değil.

  Problem 2: Karakter sayımı
    "strawberry'de kaç r var?" → LLM genellikle hata yapar
    Çünkü: "strawberry" → ["str", "awb", "erry"] gibi bölünüyor
    Model karakterlere değil tokenlara bakıyor.

  Problem 3: Dil eşitsizliği
    İngilizce: "hello" → 1 token
    Türkçe: "merhaba" → ["mer", "hab", "a"] → 3 token
    → Non-English diller daha pahalı (hem hesap hem bağlam)

  Problem 4: Başında/sonunda boşluk
    "hello" ve " hello" genellikle farklı tokenlar!
    GPT-2: "hello"=31373, " hello"=23748

  Çözüm yolları:
    - Daha büyük vocab (GPT-4: 100K, LLaMA-3: 128K)
    - Özel aritmetik tokenlar
    - Chain-of-thought ile sayısal muhakeme
    """)


if __name__ == "__main__":
    neden_tokenizasyon()
    bpe_demo()
    byte_level_bpe_ozet()
    special_tokens()
    tokenizasyon_sorunlari()
