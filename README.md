# O'zbek Tili Uchun Avtomatik Nutqni Tanish (ASR) Tizimi

## Magistrlik Dissertatsiyasi

### 📂 Loyiha Strukturasi

```
uzbek_asr_thesis/
├── data/                      # Dataset
│   ├── clean/                 # Toza audio (10 soat)
│   ├── noisy/                 # Shovqinli audio
│   └── transcripts/           # Matn transkriptlari
├── literature_review/         # Ilmiy adabiyotlar tahlili
├── models/                    # ASR modellari kodi
│   ├── mfcc_cnn_ctc/         # Model 1: MFCC + CNN + CTC
│   ├── wav2vec2/             # Model 2: Wav2Vec2 Fine-tuning
│   └── whisper/              # Model 3: Whisper (O'zbek)
├── experiments/               # Tajribalar va natijalar
├── notebooks/                 # Jupyter notebooks
├── real_time_app/            # Real-time ASR tizim
└── requirements.txt          # Python kutubxonalari
```

### 🎯 Dissertatsiya Rejalari

#### ✅ 1-BOSQICH: NAZARIY QISM (1-2 oy)

- [ ] 20+ ilmiy manba tahlili
- [ ] Qiyosiy jadval yaratish
- [ ] Matematik isbotlar tayyorlash
- [ ] Literature Review yozish (10-15 bet)

#### 🎤 2-BOSQICH: DATASET (1 oy)

- [ ] 10 soat audio yig'ish
- [ ] 5 yosh guruhi, 3 lahja
- [ ] Transkript yaratish
- [ ] Data preprocessing

#### 🤖 3-BOSQICH: MODELLAR (2-3 oy)

- [ ] Model 1: MFCC + CNN + CTC
- [ ] Model 2: Wav2Vec2 Fine-tuning
- [ ] Model 3: Whisper (O'zbek)
- [ ] Har bir model uchun training

#### 📊 4-BOSQICH: TAHLIL (1 oy)

- [ ] WER/CER hisoblash
- [ ] Shovqinga bardoshlilik testi
- [ ] Lahjalar bo'yicha tahlil
- [ ] Grafiklar va jadvallar

#### 🚀 5-BOSQICH: REAL-TIME TIZIM (1 oy)

- [ ] FastAPI/Streamlit interfeys
- [ ] Mikrofon integratsiyasi
- [ ] VAD (Voice Activity Detection)
- [ ] Demo yaratish

#### 🌟 BONUS (Agar vaqt bo'lsa)

- [ ] ONNX eksport
- [ ] Quantization
- [ ] Mobile app (Android/Flutter)
- [ ] Offline ASR

### 📚 Kerakli Ilmiy Manbalar

#### Klassik ASR:

1. Rabiner, L. (1989) - "A Tutorial on HMM and Applications in Speech Recognition"
2. Gales, M. & Young, S. (2008) - "The Application of HMM to Speech Recognition"

#### Zamonaviy ASR:

3. Graves, A. et al. (2006) - "Connectionist Temporal Classification (CTC)"
4. Baevski, A. et al. (2020) - "wav2vec 2.0"
5. Radford, A. et al. (2022) - "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)"
6. Vaswani, A. et al. (2017) - "Attention Is All You Need"

#### O'zbek Tili ASR:

7. Mamyrbayev, O. et al. - "Automatic Speech Recognition for Turkic Languages"
8. Safarov, F. et al. - "Challenges in Uzbek Speech Recognition"

**Qo'shimcha**: arXiv, IEEE Xplore, Google Scholar dan ko'proq manbalar topish kerak.

### 📊 Qiyosiy Jadval (Namuna)

| Model    | Afzalligi             | Kamchiligi     | Resurs | WER    |
| -------- | --------------------- | -------------- | ------ | ------ |
| HMM-GMM  | Kam resurs, tez       | Aniqlik past   | CPU    | 35-40% |
| CTC      | Oddiy, parallelizable | Alignment yo'q | GPU    | 15-20% |
| Wav2Vec2 | Transfer learning     | Dataset kerak  | GPU    | 10-15% |
| Whisper  | Juda aniq, ko'p til   | Og'ir, sekin   | GPU    | 5-10%  |
| RNN-T    | Online streaming      | Murakkab       | GPU    | 12-18% |

### 🧮 Matematik Formulalar

#### 1. Bayes Formulasi (ASR asosi):

$$\hat{W} = \arg\max_{W} P(X|W)P(W)$$

Bu yerda:

- $P(X|W)$ - Akustik model (audio berilgan so'z ehtimoli)
- $P(W)$ - Til modeli (so'z ketma-ketligi ehtimoli)

#### 2. CTC Loss:

$$P(y|x) = \sum_{\pi \in B^{-1}(y)} \prod_{t} P(\pi_t|x)$$

#### 3. Attention Mechanism:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### 4. WER (Word Error Rate):

$$\text{WER} = \frac{S + D + I}{N} \times 100\%$$

Bu yerda: S=substitution, D=deletion, I=insertion, N=umumiy so'zlar

### 💻 Kerakli Texnologiyalar

**Python 3.8+** va quyidagi kutubxonalar:

- **Audio**: librosa, soundfile, pydub, webrtcvad
- **Deep Learning**: torch, tensorflow, transformers
- **Feature Extraction**: python_speech_features, scipy
- **ASR Models**: wav2vec2, whisper, speechbrain
- **Evaluation**: jiwer (WER/CER)
- **Visualization**: matplotlib, seaborn, plotly
- **Web App**: fastapi, streamlit, gradio
- **Utils**: numpy, pandas, tqdm

### 📈 Kutilayotgan Natijalar

1. **Dataset**: 10 soat o'zbek nutq dataseti (Open-source)
2. **Modellar**: 3 ta ASR model
3. **WER**: ~10% dan kam (eng yaxshi model)
4. **Real-time**: <500ms latency
5. **Dissertatsiya**: 80-100 bet

### ⚠️ Muhim Ogohlantirishlar

1. **GPU kerak**: Google Colab (bepul) yoki Cloud GPU
2. **Vaqt**: Kamida 6 oy
3. **Dataset**: Eng og'ir qism, boshqalardan yordam oling
4. **Matematik**: Formulalarni tushunib yozing, ko'chirmaslik
5. **Plagiarism**: Albatta citation qo'yish

### 🎓 Himoya Uchun Tayyorgarlik

1. Har bir formulani tushuntirib bering
2. Modellarning ishlash jarayonini bilish
3. Nima uchun aynan shu modellarni tanlaganingiz
4. O'zbek tili uchun qanday muammolar bor
5. Kelajakdagi ishlar (Future Work)

---

**Muvaffaqiyatlar! 🚀**
