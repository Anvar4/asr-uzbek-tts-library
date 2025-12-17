# ILMIY ADABIYOTLAR TAHLILI (Literature Review)

## O'zbek Tili Uchun Avtomatik Nutqni Tanish Tizimi

---

## 1. KIRISH

### 1.1. Mavzuning Dolzarbligi

Avtomatik nutqni tanish (ASR - Automatic Speech Recognition) zamonaviy sun'iy intellekt sohalaridan biri...

### 1.2. O'zbek Tili Xususiyatlari

- Agglutinativ til strukturasi
- 6 ta lahja (Toshkent, Farg'ona, Xorazm, Samarqand, Jizzax, Surxondaryo)
- Fonetik xususiyatlar
- Morfologik murakkablik

---

## 2. KLASSIK ASR YONDASHUVLARI

### 2.1. HMM-GMM (Hidden Markov Model - Gaussian Mixture Model)

**Manba 1:**

- **Muallif**: Rabiner, L. R. (1989)
- **Sarlavha**: "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"
- **Nashr**: Proceedings of the IEEE, 77(2), 257-286
- **Asosiy g'oya**: HMM nutq signallarini statistik modellashtirish uchun ishlatiladi
- **Afzallik**:
  - Kam resurs talab qiladi
  - CPU da ishlaydi
  - Yaxshi nazariy asos
- **Kamchilik**:
  - Past aniqlik (~30-40% WER)
  - Feature engineering kerak
  - Murakkab til uchun cheklangan

**Manba 2:**

- **Muallif**: Gales, M., & Young, S. (2008)
- **Sarlavha**: "The Application of Hidden Markov Models in Speech Recognition"
- **Nashr**: Foundations and Trends in Signal Processing
- **Hissa**: HMM arxitekturasi va training algoritmlari
- **WER**: 35-45% (ingliz tili uchun)

---

## 3. ZAMONAVIY DEEP LEARNING YONDASHUVLARI

### 3.1. Connectionist Temporal Classification (CTC)

**Manba 3:**

- **Muallif**: Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006)
- **Sarlavha**: "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"
- **Nashr**: ICML 2006
- **Asosiy g'oya**: Alignment muammosini blank token orqali hal qilish
- **Matematik formulasi**:
  $$P(y|x) = \sum_{\pi \in B^{-1}(y)} \prod_{t} P(\pi_t|x)$$
- **Afzallik**:
  - Alignment shart emas
  - End-to-end training
  - Parallel hisoblash
- **Kamchilik**:
  - Til modeli yo'q
  - Independent frames (context cheklangan)
- **WER**: 15-25%

**Manba 4:**

- **Muallif**: Hannun, A. et al. (2014)
- **Sarlavha**: "Deep Speech: Scaling up end-to-end speech recognition"
- **Nashr**: arXiv:1412.5567
- **Model**: RNN + CTC
- **Natija**: WER 16.5% (ingliz tili)

---

### 3.2. Attention Mechanisms

**Manba 5:**

- **Muallif**: Vaswani, A. et al. (2017)
- **Sarlavha**: "Attention Is All You Need"
- **Nashr**: NeurIPS 2017
- **Asosiy g'oya**: Self-attention orqali global context
- **Formulasi**:
  $$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **ASR ga ta'siri**: Transformer arxitekturasi ASR uchun yangi davr
- **Afzallik**:
  - Uzoq masofa dependencies
  - Parallel processing
  - Scalable

**Manba 6:**

- **Muallif**: Bahdanau, D., Cho, K., & Bengio, Y. (2015)
- **Sarlavha**: "Neural Machine Translation by Jointly Learning to Align and Translate"
- **Nashr**: ICLR 2015
- **Hissa**: Encoder-Decoder attention ASR ga qo'llash

---

### 3.3. Wav2Vec 2.0

**Manba 7:**

- **Muallif**: Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020)
- **Sarlavha**: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
- **Nashr**: NeurIPS 2020
- **Asosiy g'oya**:
  - Self-supervised pre-training
  - Contrastive learning
  - Fine-tuning kam data bilan
- **Arxitektura**:
  - CNN feature encoder
  - Transformer contextualized representations
  - Quantization module
- **Natija**:
  - 10 minut labeled data bilan WER 4.8%
  - Low-resource tillar uchun juda samarali
- **O'zbek tili uchun**: Transfer learning imkoniyati

---

### 3.4. Whisper

**Manba 8:**

- **Muallif**: Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022)
- **Sarlavha**: "Robust Speech Recognition via Large-Scale Weak Supervision"
- **Nashr**: OpenAI Technical Report
- **Dataset**: 680,000 soat (99 til)
- **Arxitektura**:
  - Encoder-Decoder Transformer
  - Multitask learning (transcription, translation, VAD)
- **Model o'lchamlari**: tiny (39M) → large (1550M)
- **Natija**:
  - SOTA ko'plab benchmarks da
  - Shovqinga bardoshli
  - Ko'p til (zero-shot)
- **O'zbek tili**: Limited support, fine-tuning kerak

---

## 4. FEATURE EXTRACTION

### 4.1. MFCC (Mel-Frequency Cepstral Coefficients)

**Manba 9:**

- **Muallif**: Davis, S., & Mermelstein, P. (1980)
- **Sarlavha**: "Comparison of Parametric Representations for Monosyllabic Word Recognition"
- **Nashr**: IEEE Transactions on ASSP
- **Jarayon**:
  1. Pre-emphasis
  2. Framing
  3. Windowing (Hamming)
  4. FFT
  5. Mel-filter banks
  6. Log
  7. DCT
- **Afzallik**:
  - Inson eshitish tizimini taqlid qiladi
  - Kam o'lcham (13-40 coefficients)
- **Kamchilik**:
  - Shovqinga sezgir
  - Phase information yo'qoladi

### 4.2. Mel-Spectrogram

**Manba 10:**

- **Muallif**: Various studies (2015-2020)
- **Afzallik**:
  - Neural networks uchun yaxshi
  - 2D image kabi ishlash
  - Time-frequency representation
- **Qo'llanish**:
  - CNN models
  - Deep learning ASR

---

## 5. O'ZBEK TILI UCHUN ASR TADQIQOTLARI

### 5.1. Turkik Tillar ASR

**Manba 11:**

- **Muallif**: Mamyrbayev, O., Alimhan, K., Oralbekova, D., & Bekarystankyzy, A. (2020)
- **Sarlavha**: "Automatic Speech Recognition for Turkic Languages"
- **Nashr**: MDPI Information
- **Tillar**: Qozoq, O'zbek, Qirg'iz, Turkman
- **Muammolar**:
  - Dataset tanqisligi
  - Morfologik murakkablik
  - Lahja xilma-xilligi
- **Yechim**: Transfer learning bilan multilingual model

**Manba 12:**

- **Muallif**: Safarov, F., Temurbek, K., & Cho, Y. I. (2022)
- **Sarlavha**: "Uzbek Speech Recognition using Deep Learning"
- **Dataset**: 100 soat o'zbek nutq
- **Model**: Wav2Vec2 + CTC
- **WER**: 18.5%
- **Muammolar**:
  - Toza dataset yo'qligi
  - Lahja variatsiyasi
  - Code-switching (rus, ingliz so'zlari)

---

## 6. SHOVQINGA BARDOSHLILIK

### 6.1. Noise Reduction Algorithms

**Manba 13:**

- **Muallif**: Loizou, P. C. (2013)
- **Sarlavha**: "Speech Enhancement: Theory and Practice"
- **Usullar**:
  - **Spectral Subtraction**: Shovqin spektrini ayirish
  - **Wiener Filter**: Optimal filtering
  - **Deep Noise Suppression**: Neural network based

**Manba 14:**

- **Muallif**: Reddy, C. K. et al. (2020)
- **Sarlavha**: "INTERSPEECH 2020 Deep Noise Suppression Challenge"
- **Yechim**: RNN-based real-time denoising
- **Natija**: 35% WER reduction shovqinli muhitda

---

## 7. EVALUATION METRICS

### 7.1. WER (Word Error Rate)

**Formula**:
$$\text{WER} = \frac{S + D + I}{N} \times 100\%$$

- S = Substitutions (o'rniga boshqa so'z)
- D = Deletions (so'z tushib qolgan)
- I = Insertions (qo'shimcha so'z)
- N = Umumiy so'zlar

**Manba 15:**

- **Muallif**: Klakow, D., & Peters, J. (2002)
- **Sarlavha**: "Testing the Correlation of Word Error Rate and Perceptual Quality in Speech Recognition"
- **Xulosa**: WER har doim perceptual quality ni aks ettirmaydi

### 7.2. CER (Character Error Rate)

O'zbek tili kabi agglutinativ tillar uchun CER ko'proq ma'noli:
$$\text{CER} = \frac{S_c + D_c + I_c}{N_c} \times 100\%$$

---

## 8. REAL-TIME ASR

### 8.1. Streaming Models

**Manba 16:**

- **Muallif**: He, Y. et al. (2019)
- **Sarlavha**: "Streaming End-to-end Speech Recognition for Mobile Devices"
- **Talablar**:
  - Latency < 300ms
  - Real-time factor < 0.1
  - Memory efficient

**Manba 17:**

- **Muallif**: RNN-Transducer approach
- **Afzallik**:
  - Online streaming
  - No future context needed
- **Qo'llanish**: Google Assistant, Siri

---

## 9. QIYOSIY JADVAL

| Model     | Yil       | Arxitektura     | WER (ingliz) | Parameters | Resurs |
| --------- | --------- | --------------- | ------------ | ---------- | ------ |
| HMM-GMM   | 1980-2010 | Statistical     | 35-45%       | ~10M       | CPU    |
| DNN-HMM   | 2010-2015 | Hybrid          | 20-30%       | ~100M      | GPU    |
| CTC       | 2015-2018 | RNN/CNN         | 15-25%       | ~50M       | GPU    |
| Attention | 2017-2020 | Seq2Seq         | 10-20%       | ~100M      | GPU    |
| Wav2Vec2  | 2020      | Self-supervised | 5-10%        | 300M       | GPU    |
| Whisper   | 2022      | Transformer     | 3-8%         | 1550M      | GPU    |

---

## 10. O'ZBEK TILI UCHUN XULOSA VA TAVSIYALAR

### 10.1. Asosiy Muammolar

1. **Dataset tanqisligi**: Open-source o'zbek nutq dataseti yo'q
2. **Lahja xilma-xilligi**: 6 ta asosiy lahja
3. **Morfologik murakkablik**: Agglutinativ struktura
4. **Code-switching**: Rus va ingliz so'zlari aralashmasi
5. **Shovqinli muhit**: Real-world scenarios

### 10.2. Tavsiya Etiladigan Yondashuv

1. **Dataset yaratish**: Kamida 10-20 soat (dissertation uchun)
2. **Transfer learning**: Wav2Vec2 yoki Whisper fine-tuning
3. **CTC Loss**: End-to-end training uchun
4. **Data augmentation**:
   - Speed perturbation
   - Noise injection
   - SpecAugment
5. **Evaluation**: WER va CER ikkalasi ham

---

## 11. KELAJAK YO'NALISHLARI (Future Work)

**Manba 18-20:**

- **Low-resource ASR**: Few-shot learning
- **Multimodal**: Audio + Text + Video
- **Edge Computing**: Mobile ASR
- **Personalization**: User-specific adaptation
- **Dialect-aware models**: Lahja tanish + ASR

---

## 12. REFERENCES

[1] Rabiner, L. R. (1989). A tutorial on hidden Markov models...

[2] Graves, A., et al. (2006). Connectionist temporal classification...

[3] Vaswani, A., et al. (2017). Attention is all you need...

[4] Baevski, A., et al. (2020). wav2vec 2.0...

[5] Radford, A., et al. (2022). Robust speech recognition...

... (20+ references)

---

**Eslatma**: Har bir manba uchun:

- To'liq citation (APA yoki IEEE format)
- PDF link (agar mavjud bo'lsa)
- Asosiy formulalar va grafiklar
- O'z dissertatsiyangizga qanday bog'liqligini yozing

---

## 📝 Qanday Yozish Kerak:

1. **Har bir manba uchun**:

   - Nima qilgan?
   - Qaysi model/algoritm?
   - Natijalar (WER, CER)?
   - Afzallik va kamchiliklar?
   - O'z ishingizga qanday bog'liq?

2. **Critical Analysis**:

   - Faqat ko'chirmaslik!
   - Taqqoslash
   - O'z fikringiz

3. **Visualization**:
   - Jadvallar
   - Grafiklar (WER evolution over years)
   - Arxitektura rasmlari

---

**Muvaffaqiyatlar! 📚🎓**
