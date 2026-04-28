"""
=============================================================
MODÜL 1.4 — BİLGİ TEORİSİ (Information Theory)
=============================================================

LLM'lerde neden bilgi teorisi?
  - Cross-entropy loss → dil modelinin eğitim hedefi
  - Perplexity → dil modelinin değerlendirme metriği
  - KL divergence → RLHF'de PPO, VAE, model distillation
  - Mutual information → feature selection, probing

Konular:
  1. Bilgi içeriği (self-information)
  2. Shannon Entropisi
  3. Cross-entropy
  4. KL Divergence
  5. Mutual Information
  6. Perplexity — LLM değerlendirme metriği
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. BİLGİ İÇERİĞİ (Self-Information)
# ─────────────────────────────────────────────────────────────
# Bir olayın "şaşırtıcılığı" (ne kadar az bekleniyor):
#
#   I(x) = -log₂ P(x)   [bit cinsinden]
#   I(x) = -ln P(x)      [nat cinsinden — ML'de genellikle ln kullanılır]
#
# Özellikler:
#   P(x) = 1   → I(x) = 0   (kesin olay, bilgi taşımaz)
#   P(x) = 0.5 → I(x) = 1 bit
#   P(x) → 0  → I(x) → ∞  (çok nadir olay, çok bilgi taşır)

def bilgi_icerigi():
    print("=" * 55)
    print("1. BİLGİ İÇERİĞİ (Self-Information)")
    print("=" * 55)

    probabilities = [1.0, 0.5, 0.25, 0.1, 0.01, 0.001]
    print(f"{'P(x)':>8}  {'I(x) [bit]':>12}  {'I(x) [nat]':>12}")
    print("-" * 38)
    for p in probabilities:
        I_bit = -np.log2(p)
        I_nat = -np.log(p)
        print(f"{p:>8.3f}  {I_bit:>12.4f}  {I_nat:>12.4f}")

    # LLM bağlantısı:
    # "the" kelimesi çok sık → düşük I → modelin öğrenmesi kolay
    # "serendipity" çok nadir → yüksek I → modelin doğru tahmin etmesi zor
    print("\nLLM örneği:")
    p_the = 0.05       # "the" kelimesi corpus'ta ~%5
    p_rare = 0.00001   # nadir teknik terim
    print(f"  P('the')          = {p_the}   → I = {-np.log(p_the):.2f} nat")
    print(f"  P('serendipity')  = {p_rare} → I = {-np.log(p_rare):.2f} nat")


# ─────────────────────────────────────────────────────────────
# 2. SHANNON ENTROPİSİ
# ─────────────────────────────────────────────────────────────
# Bir dağılımın ortalama bilgi içeriği (belirsizliği):
#
#   H(P) = -Σ_x P(x) log P(x)  =  E_P[-log P(X)]
#
# Özellikler:
#   H(P) ≥ 0
#   H maksimum → P uniform dağılım  (maksimum belirsizlik)
#   H = 0 → P deterministic  (sıfır belirsizlik)
#
# LLM bağlantısı:
#   Dil modelinin çıktı entropisi → ne kadar "emin" olduğu
#   Yüksek entropi → model emin değil, birçok seçenek mümkün

def shannon_entropisi():
    print("\n" + "=" * 55)
    print("2. SHANNON ENTROPİSİ")
    print("=" * 55)

    def entropy(p, base=np.e):
        p = np.array(p, dtype=float)
        p = p[p > 0]  # log(0) tanımsız, 0*log(0) = 0 kabul et
        if base == 2:
            return -np.sum(p * np.log2(p))
        return -np.sum(p * np.log(p))

    # Çeşitli dağılımlar
    print(f"{'Dağılım':30s}  {'H(P) [nat]':>12}  {'H(P) [bit]':>12}")
    print("-" * 58)

    p_uniform_4 = [0.25, 0.25, 0.25, 0.25]
    p_skewed    = [0.97, 0.01, 0.01, 0.01]
    p_two       = [0.5, 0.5]
    p_determ    = [1.0, 0.0, 0.0, 0.0]
    p_mixed     = [0.4, 0.3, 0.2, 0.1]

    cases = [
        ("Uniform [4]",       p_uniform_4),
        ("Skewed [97/1/1/1]", p_skewed),
        ("Binary [0.5,0.5]",  p_two),
        ("Deterministic",     p_determ),
        ("Mixed [4/3/2/1]",   p_mixed),
    ]
    for name, p in cases:
        H_nat = entropy(p, base=np.e)
        H_bit = entropy(p, base=2)
        print(f"{name:30s}  {H_nat:>12.4f}  {H_bit:>12.4f}")

    # Maksimum entropi teoremi: H maksimum = log(|X|) (uniform'da)
    n = 4
    H_max_bit = np.log2(n)
    print(f"\nMaksimum H ({n} sınıf) = log₂({n}) = {H_max_bit:.4f} bit")

    # LLM context: temperature ile entropi değişimi
    print("\nTemperature → Entropi ilişkisi:")
    logits = np.array([2.0, 1.0, 0.5, 3.0, -1.0])
    for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
        z = logits / T
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        H = entropy(p)
        print(f"  T={T:.1f}: H = {H:.4f} nat  (probs: {p})")


# ─────────────────────────────────────────────────────────────
# 3. CROSS-ENTROPY
# ─────────────────────────────────────────────────────────────
# İki dağılım P (gerçek) ve Q (tahmin) arasındaki cross-entropy:
#
#   H(P, Q) = -Σ_x P(x) log Q(x)  =  E_P[-log Q(X)]
#
# Ayrıştırma:
#   H(P, Q) = H(P) + KL(P || Q)
#   → H(P, Q) ≥ H(P),  eşitlik P=Q'da
#
# LLM LOSS = Cross-Entropy!
# One-hot label için (tek doğru token t*):
#   P(t) = 1[t == t*]
#   H(P, Q) = -log Q(t*)
#   → Doğru tokenin log-olasılığının negatifi!

def cross_entropy():
    print("\n" + "=" * 55)
    print("3. CROSS-ENTROPY — LLM LOSS")
    print("=" * 55)

    def cross_entropy_fn(p_true, q_pred):
        p = np.array(p_true, dtype=float)
        q = np.array(q_pred, dtype=float)
        q = np.clip(q, 1e-12, 1.0)  # log(0) önle
        return -np.sum(p * np.log(q))

    def softmax(z):
        z = np.array(z, dtype=float)
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    # Örnek: 5-token vocab, gerçek token = index 3
    vocab_size = 5
    true_token = 3

    # One-hot P
    P = np.zeros(vocab_size)
    P[true_token] = 1.0

    # Model logitleri → softmax → Q
    logits_iyi  = np.array([0.1, 0.2, 0.1, 3.0, 0.1])   # güçlü sinyal
    logits_orta = np.array([0.5, 0.5, 0.5, 1.0, 0.5])   # zayıf sinyal
    logits_kötü = np.array([2.0, 1.0, 0.5, 0.1, 1.5])   # yanlış token öngörüyor

    print(f"Gerçek token: index {true_token}")
    print(f"{'Model':10s}  {'Q[3]':>8}  {'CE Loss':>10}")
    print("-" * 32)
    for name, logits in [("İyi", logits_iyi), ("Orta", logits_orta), ("Kötü", logits_kötü)]:
        Q = softmax(logits)
        loss = cross_entropy_fn(P, Q)
        print(f"{name:10s}  {Q[true_token]:>8.4f}  {loss:>10.4f}")

    # Önemli: one-hot için CE = -log(Q[t*])
    print("\nDikkat: one-hot için CE(P,Q) = -log(Q[t*])")
    Q = softmax(logits_iyi)
    print(f"  -log(Q[{true_token}]) = {-np.log(Q[true_token]):.4f}  ≡  CE = {cross_entropy_fn(P, Q):.4f}")

    # Sekans için ortalama CE loss
    print("\n--- Sekans Loss (token başına ortalama) ---")
    # "Hello world" → token ids [15496, 995]
    # Her adımda model bir token tahmin eder
    sequence_logits = [
        np.array([1.0, 3.0, 0.5, 0.2]),  # adım 1: true_token=1
        np.array([0.2, 0.5, 2.5, 0.1]),  # adım 2: true_token=2
        np.array([0.1, 0.1, 0.1, 3.5]),  # adım 3: true_token=3
    ]
    true_tokens = [1, 2, 3]

    losses = []
    for logits, t in zip(sequence_logits, true_tokens):
        q = softmax(logits)
        loss = -np.log(q[t])
        losses.append(loss)
        print(f"  Adım {t}: q[{t}]={q[t]:.4f}, loss={loss:.4f}")

    avg_loss = np.mean(losses)
    print(f"  Ortalama CE loss: {avg_loss:.4f}")
    print(f"  Perplexity: exp({avg_loss:.4f}) = {np.exp(avg_loss):.4f}")


# ─────────────────────────────────────────────────────────────
# 4. KL DIVERGENCE
# ─────────────────────────────────────────────────────────────
# P'den Q'ya KL divergence (relative entropy):
#
#   KL(P || Q) = Σ_x P(x) log [P(x) / Q(x)]
#             = H(P, Q) - H(P)
#             = E_P[log P(X) - log Q(X)]
#
# Özellikler:
#   KL(P || Q) ≥ 0        (Gibbs eşitsizliği)
#   KL(P || Q) = 0 ↔ P = Q
#   KL(P || Q) ≠ KL(Q || P)  (simetrik değil!)
#
# LLM bağlantıları:
#   - RLHF (PPO): KL cezası ile politika değişimini sınırla
#   - DPO: KL(π_θ || π_ref) cezası
#   - Knowledge Distillation: KL(P_teacher || P_student)
#   - VAE: ELBO = E[log p] - KL(q || p)

def kl_divergence():
    print("\n" + "=" * 55)
    print("4. KL DIVERGENCE")
    print("=" * 55)

    def kl_div(P, Q):
        P = np.array(P, dtype=float)
        Q = np.array(Q, dtype=float)
        Q = np.clip(Q, 1e-12, 1.0)
        P = np.clip(P, 1e-12, 1.0)
        return np.sum(P * np.log(P / Q))

    P = np.array([0.4, 0.3, 0.2, 0.1])   # "gerçek" dağılım
    Q = np.array([0.25, 0.25, 0.25, 0.25])  # uniform tahmin
    R = np.array([0.5, 0.3, 0.1, 0.1])   # P'ye yakın tahmin

    print(f"P = {P}")
    print(f"Q (uniform) = {Q}")
    print(f"R (P'ye yakın) = {R}")
    print(f"\nKL(P || Q) = {kl_div(P, Q):.4f}  (uniform'dan uzak)")
    print(f"KL(P || R) = {kl_div(P, R):.4f}  (P'ye yakın, daha küçük)")
    print(f"KL(Q || P) = {kl_div(Q, P):.4f}  (KL simetrik değil!)")

    # RLHF KL cezası örneği
    print("\n--- RLHF KL Cezası ---")
    # Reference model (SFT): π_ref
    # Fine-tuned model: π_θ
    # RLHF hedefi: max E[r(x,y)] - β * KL(π_θ || π_ref)
    beta = 0.1
    pi_ref = np.array([0.3, 0.2, 0.3, 0.2])  # reference policy
    pi_theta_iyi   = np.array([0.4, 0.2, 0.3, 0.1])  # az değişim
    pi_theta_kötü  = np.array([0.7, 0.1, 0.1, 0.1])  # çok değişim

    r_iyi  = 0.8  # reward
    r_kötü = 1.2  # yüksek reward ama çok fazla değişti

    for name, pi, r in [("Az değişim", pi_theta_iyi, r_iyi),
                         ("Çok değişim", pi_theta_kötü, r_kötü)]:
        kl = kl_div(pi, pi_ref)
        objective = r - beta * kl
        print(f"  {name}: r={r}, KL={kl:.4f}, hedef={objective:.4f}")


# ─────────────────────────────────────────────────────────────
# 5. MUTUAL INFORMATION
# ─────────────────────────────────────────────────────────────
# X ve Y arasındaki paylaşılan bilgi:
#
#   I(X; Y) = Σ_{x,y} P(x,y) log [P(x,y) / (P(x)P(y))]
#           = H(X) - H(X|Y) = H(Y) - H(Y|X)
#           = H(X) + H(Y) - H(X,Y)
#
# I(X;Y) = 0 ↔ X ve Y bağımsız
# I(X;Y) = H(X) ↔ Y, X'i tamamen belirliyor
#
# LLM bağlantısı:
#   Attention: token'lar arası mutual information yüksek → güçlü dikkat ağırlığı

def mutual_information():
    print("\n" + "=" * 55)
    print("5. MUTUAL INFORMATION")
    print("=" * 55)

    def entropy(p):
        p = np.array(p, dtype=float)
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def mutual_info(joint_prob):
        joint = np.array(joint_prob, dtype=float)
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        I = 0.0
        for i in range(joint.shape[0]):
            for j in range(joint.shape[1]):
                if joint[i,j] > 0:
                    I += joint[i,j] * np.log(joint[i,j] / (px[i] * py[j]))
        return I

    # Örnek 1: Bağımlı değişkenler
    joint_dependent = np.array([
        [0.3, 0.1],
        [0.1, 0.5]
    ])
    # Örnek 2: Bağımsız değişkenler (P(x,y) = P(x)*P(y))
    joint_independent = np.array([
        [0.2, 0.3],
        [0.2, 0.3]
    ])
    # Her ikisini normalleştir
    joint_dependent /= joint_dependent.sum()
    joint_independent /= joint_independent.sum()

    I_dep   = mutual_info(joint_dependent)
    I_indep = mutual_info(joint_independent)
    print(f"Bağımlı değişkenler:  I(X;Y) = {I_dep:.4f}")
    print(f"Bağımsız değişkenler: I(X;Y) = {I_indep:.6f}  (≈ 0)")


# ─────────────────────────────────────────────────────────────
# 6. PERPLEXITY — LLM DEĞERLENDİRME METRİĞİ
# ─────────────────────────────────────────────────────────────
# N token'lık bir sekans için:
#
#   PPL = exp(-1/N * Σ_{t=1}^{N} log P(w_t | w_{1:t-1}))
#       = exp(CE_loss)
#
# Yorumlama:
#   PPL = 10  → model her adımda ~10 eşit seçenek arasında kararsız
#   PPL = 1   → model her token'ı tam kesinlikle tahmin ediyor
#   PPL → ∞   → model çok kötü
#
# GPT-2 (1.5B): Penn Treebank PPL ≈ 35
# GPT-3 (175B): Penn Treebank PPL ≈ 20

def perplexity():
    print("\n" + "=" * 55)
    print("6. PERPLEXITY — LLM DEĞERLENDİRME METRİĞİ")
    print("=" * 55)

    def compute_perplexity(log_probs):
        # log_probs: her adımda doğru tokenin log olasılığı
        N = len(log_probs)
        avg_neg_log_prob = -np.mean(log_probs)
        ppl = np.exp(avg_neg_log_prob)
        return ppl, avg_neg_log_prob

    # İyi model: doğru tokenlara yüksek olasılık
    log_probs_good = np.log([0.8, 0.7, 0.9, 0.85, 0.75])
    # Kötü model: çok düşük olasılıklar
    log_probs_bad  = np.log([0.1, 0.05, 0.2, 0.08, 0.15])
    # Mükemmel model: her zaman doğru
    log_probs_perfect = np.log([1.0, 1.0, 1.0, 1.0, 1.0])

    print(f"{'Model':12s}  {'Ort. CE':>10}  {'PPL':>10}")
    print("-" * 36)
    for name, lp in [("İyi", log_probs_good), ("Kötü", log_probs_bad), ("Mükemmel", log_probs_perfect)]:
        ppl, ce = compute_perplexity(lp)
        print(f"{name:12s}  {ce:>10.4f}  {ppl:>10.4f}")

    # Gerçek LLM değerleri
    print("\n--- Gerçek LLM PPL Değerleri (Penn Treebank) ---")
    models = [
        ("LSTM (2018)",   35.0),
        ("Transformer-XL", 21.8),
        ("GPT-2 (1.5B)",  35.8),
        ("GPT-3 (175B)",  20.5),
    ]
    for name, ppl in models:
        ce = np.log(ppl)
        print(f"  {name:25s}: PPL={ppl:6.1f}  (CE ≈ {ce:.3f} nat)")

    print("\nNot: PPL = exp(CE_loss), bu yüzden training loss ile doğrudan karşılaştırılabilir.")


if __name__ == "__main__":
    bilgi_icerigi()
    shannon_entropisi()
    cross_entropy()
    kl_divergence()
    mutual_information()
    perplexity()
