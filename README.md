# LLM Education — Sıfırdan LLM Researcher

Bu repo, LLM araştırmacısı olmak isteyen biri için tasarlanmış adım adım bir müfredattır.
Her dosya: **matematiksel türetim + NumPy implementasyonu + PyTorch implementasyonu** içerir.

## Müfredat Haritası

```
Modül 1 → Matematiksel Temeller      (lineer cebir, olasılık, kalkülüs, bilgi teorisi)
Modül 2 → Sinir Ağları               (perceptron, backprop, aktivasyonlar, loss)
Modül 3 → Attention Mekanizması      (scaled dot-product, multi-head, positional encoding)
Modül 4 → Transformer Mimarisi       (LayerNorm, FFN, encoder, decoder, tam transformer)
Modül 5 → Dil Modellemesi            (BPE tokenization, GPT sıfırdan, eğitim döngüsü)
Modül 6 → Eğitim Teknikleri          (optimizerlar, LR schedules, gradient tricks)
Modül 7 → Modern Mimariler           (GPT-2, BERT, LLaMA farklılıkları)
```

## Nasıl Kullanılır

```bash
pip install -r requirements.txt

# Herhangi bir dosyayı çalıştır:
python 01_math_foundations/01_linear_algebra.py
python 03_attention_mechanism/01_scaled_dot_product.py
```

## Öğrenme Sırası

| Adım | Dosya | Öğrenilen Kavram |
|------|-------|-----------------|
| 1 | `01_math_foundations/01_linear_algebra.py` | Vektör uzayları, matris çarpımı, SVD |
| 2 | `01_math_foundations/02_probability_statistics.py` | Olasılık, Bayes, MLE, softmax |
| 3 | `01_math_foundations/03_calculus_autodiff.py` | Türev, gradyan, zincir kuralı |
| 4 | `01_math_foundations/04_information_theory.py` | Entropi, cross-entropy, KL div |
| 5 | `02_neural_networks/01_perceptron_mlp.py` | Perceptron, ileri geçiş |
| 6 | `02_neural_networks/02_backpropagation.py` | Geri yayılım türetimleri |
| 7 | `02_neural_networks/03_activation_functions.py` | ReLU, GELU, SiLU |
| 8 | `02_neural_networks/04_loss_functions.py` | Cross-entropy, NLL |
| 9 | `03_attention_mechanism/01_scaled_dot_product.py` | **Attention formülü** |
| 10 | `03_attention_mechanism/02_multi_head_attention.py` | Multi-head attention |
| 11 | `03_attention_mechanism/03_positional_encoding.py` | Sinüzoidal PE, RoPE |
| 12 | `04_transformer/01_layer_norm_ffn.py` | LayerNorm, FFN |
| 13 | `04_transformer/02_encoder_block.py` | Transformer encoder |
| 14 | `04_transformer/03_decoder_block.py` | Transformer decoder |
| 15 | `04_transformer/04_full_transformer.py` | Tam transformer |
| 16 | `05_language_modeling/01_tokenization_bpe.py` | BPE tokenization |
| 17 | `05_language_modeling/02_gpt_from_scratch.py` | GPT modeli |
| 18 | `05_language_modeling/03_training_loop.py` | Eğitim + perplexity |
| 19 | `06_training_techniques/01_optimizers.py` | SGD, Adam, AdamW |
| 20 | `06_training_techniques/02_lr_schedules.py` | Warmup, cosine decay |
| 21 | `06_training_techniques/03_gradient_tricks.py` | Clipping, mixed precision |
| 22 | `07_modern_architectures/01_gpt2_vs_original.py` | GPT-2 değişiklikleri |
| 23 | `07_modern_architectures/02_bert_architecture.py` | BERT, MLM |
| 24 | `07_modern_architectures/03_llama_architecture.py` | RMSNorm, RoPE, SwiGLU |
