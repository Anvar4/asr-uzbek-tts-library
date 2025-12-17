# 🎤 O'zbek Nutqni Tanish Tizimi - Loyiha Haqida

## 📋 Loyiha Nima?

Bu loyiha **O'zbek tilida avtomatik nutqni tanish (ASR - Automatic Speech Recognition)** tizimi yaratish uchun mo'ljallangan ilmiy-tadqiqot ishidir. 

## 🎯 Asosiy Maqsad

**Muammo:** O'zbek tili uchun ochiq manbali, sifatli nutqni tanish tizimlari juda kam. Google, Yandex kabi kompaniyalar yopiq tizimlarni taklif qiladi, lekin ular:

- Ochiq kodli emas (open source emas)
- O'zbek lahjalari va dialektlarini yaxshi tanimaydi
- Shovqinli muhitda yomon ishlaydi
- Mahalliy ilmiy tadqiqotlar uchun yaroqsiz

**Yechim:** Ushbu loyiha quyidagilarni amalga oshiradi:

1. **10 soatlik o'zbek nutq dataseti** yaratish (turli yosh, jins, lahja bilan)
2. **3 xil zamonaviy ASR modelini** taqqoslash va o'rgatish
3. **Shovqinga chidamli** tizim yaratish
4. **Real vaqtda ishlash** imkoniyati
5. **Ilmiy tadqiqot** - WER/CER metrikalari, matematik dalillar

## 🔬 Ilmiy Ahamiyati

### Nima Uchun Bu Loyiha Kerak?

1. **O'zbek tili uchun mahalliy yechim**

   - Xorijiy kompaniyalarga bog'liq bo'lmaslik
   - Lahja va dialektlarni to'g'ri tanlash
   - Ma'lumotlar xavfsizligi (local processing)

2. **Ilmiy tadqiqot uchun asos**

   - Kamida 20 ta ilmiy manbaga asoslangan
   - Matematik dalillar (CTC loss, attention mechanism)
   - WER (Word Error Rate) va CER (Character Error Rate) tahlili
   - Turli model arxitekturalarini taqqoslash

3. **Amaliy qo'llash**
   - Ovozli yordamchi (voice assistant) yaratish
   - Video subtitrlash avtomatlash
   - Call center yozuvlarini tahlil qilish
   - Nogironlar uchun yordamchi texnologiyalar

## 🛠️ Ishlatilgan Texnologiyalar

### 1. Python 3.13 - Asosiy Dasturlash Tili

**Nima uchun Python?**

- AI/ML sohasida eng mashhur til
- Katta kutubxonalar ekotizimi
- Oson o'rganish va tez prototiplash
- Google, OpenAI, Meta - hammasi Python ishlatadi

### 2. PyTorch 2.9 - Deep Learning Framework

**Nima uchun PyTorch?**

- Ilmiy tadqiqotlarda #1 framework (90% AI maqolalar)
- Dynamic computation graph - debugging oson
- GPU acceleration - tez o'rgatish
- Hugging Face bilan yaxshi integratsiya
- Facebook AI Research tomonidan qo'llab-quvvatlanadi

**Alternatiyalar:**

- ❌ TensorFlow - murakkab, statik graph
- ❌ JAX - yangi, kam community support
- ✅ PyTorch - moslashuvchan, katta community

### 3. Hugging Face Transformers 4.57 - Pre-trained Modellar

**Nima uchun Transformers?**

- 100,000+ pre-trained model
- Wav2Vec2 va Whisper uchun rasmiy kutubxona
- DataCollator - dataset yuklash oson
- Trainer API - faqat 10 qator kod bilan training

**Model arxitekturalari:**

- Wav2Vec2 (Meta/Facebook) - self-supervised learning
- Whisper (OpenAI) - 680,000 soat audio data bilan o'rgatilgan

### 4. Librosa 0.11 - Audio Processing

**Nima uchun Librosa?**

- MFCC (Mel-Frequency Cepstral Coefficients) chiqarish
- Audio augmentation (speed, pitch, noise)
- Spectrogram visualization
- Ilmiy standart - barcha ASR maqolalarda ishlatiladi

**Asosiy funksiyalar:**

- `librosa.feature.mfcc()` - akustik xususiyatlar
- `librosa.effects.pitch_shift()` - pitch augmentation
- `librosa.effects.time_stretch()` - speed augmentation

### 5. FastAPI + Uvicorn - Web Backend

**Nima uchun FastAPI?**

- Async/await - yuqori performance
- Automatic OpenAPI documentation
- Type hints bilan validation
- Flask ga qaraganda 2-3 marta tezroq

**Vazifasi:**

- Audio fayllarni qabul qilish (POST /upload)
- Metadata saqlash (speaker_id, age, gender, dialect)
- Clean/Noisy papkaga routing

### 6. HTML5 + MediaRecorder API - Frontend

**Nima uchun HTML5?**

- Cross-platform - barcha browserda ishlaydi
- MediaRecorder API - mikrofon uchun native support
- Mobil va desktopda bir xil ishlaydi
- Qo'shimcha dastur o'rnatish kerak emas

**Zamonaviy CSS:**

- CSS Grid - responsive layout
- clamp() - fluid typography
- Gradients - zamonaviy dizayn
- Animations - yaxshi UX

## 📊 Dataset Yaratish Strategiyasi

### Dataset Tuzilishi

```
data/
├── clean/          # Tinch muhitda yozilgan
│   ├── SP001_15-25_m_tashkent_001.wav
│   ├── SP001_15-25_m_tashkent_002.wav
│   └── ...
├── noisy/          # Shovqinli muhitda yozilgan
│   ├── SP002_26-35_f_fergana_001.wav
│   └── ...
└── transcripts/    # Matnli transkripsiyalar
    ├── SP001_15-25_m_tashkent_001.txt
    └── ...
```

### Metadata Parametrlari

1. **Speaker ID** - Har bir spiker uchun unique identifier
2. **Age Group** - 15-25, 26-35, 36-45, 46-55, 56+
3. **Gender** - m (erkak), f (ayol)
4. **Dialect** - tashkent, fergana, khorezm, samarkand, other
5. **Environment** - clean/noisy

### Nima Uchun Bu Parametrlar?

- **Yosh** - ovoz tovushi yoshga bog'liq o'zgaradi
- **Jins** - erkak/ayol ovozi farq qiladi (pitch, formants)
- **Lahja** - o'zbek tilida 5+ dialekt bor
- **Shovqin** - real hayotda perfect conditions yo'q

## 🧠 Uchta ASR Model Arxitekturasi

### Model 1: MFCC + CNN + LSTM + CTC (5.1M parametr)

**Arxitektura:**

```
Audio → MFCC (13 koeffitsiyent) →
CNN (3 layers, 64-128-256 channels) →
BiLSTM (256 units, 2 layers) →
Fully Connected →
CTC Loss
```

**Nima uchun bu arxitektura?**

- **MFCC** - klassik akustik xususiyatlar (1980-lardan beri)
- **CNN** - lokal pattern recognition (konsonant, vokallar)
- **BiLSTM** - temporal dependencies (so'zlar ketma-ketligi)
- **CTC Loss** - alignment-free training (text-audio moslashtirish kerak emas)

**Afzalliklari:**

- ✅ Kichik model (5MB)
- ✅ CPU da tez ishlaydi
- ✅ Matematik jihatdan tushunish oson

**Kamchiliklari:**

- ❌ Pre-trained emas (zero knowledge)
- ❌ Kam data bilan yomon o'rganadi

### Model 2: Wav2Vec2 (300M parametr)

**Arxitektura:**

```
Raw Audio →
CNN Feature Encoder →
Transformer (12 layers) →
Contrastive Learning →
Fine-tuning with CTC
```

**Nima uchun Wav2Vec2?**

- **Self-supervised learning** - labelsiz audio bilan o'rganadi
- **Pre-trained on 960h Librispeech** - English knowledge transfer
- **Feature learning** - MFCC dan yaxshiroq xususiyatlar
- **SOTA results** - WER 1.8% (LibriSpeech clean)

**Afzalliklari:**

- ✅ Transfer learning - kam data bilan yaxshi
- ✅ Phonetic representations - universal features
- ✅ Fine-tuning oson (1-2 soat GPU)

**Kamchiliklari:**

- ❌ Katta model (1.2GB)
- ❌ Faqat GPU da tez ishlaydi

### Model 3: Whisper (1.5B parametr - Large model)

**Arxitektura:**

```
Raw Audio →
Log-Mel Spectrogram →
Encoder (Transformer) →
Decoder (Transformer) →
Autoregressive Generation
```

**Nima uchun Whisper?**

- **Multilingual** - 99 til, o'zbek ham bor
- **Multitask** - transcription, translation, language detection
- **Robust** - shovqinga chidamli
- **680,000 soat audio** - eng katta dataset

**Afzalliklari:**

- ✅ Eng yaxshi accuracy
- ✅ Punctuation va capitalization
- ✅ Code-switching (rus-o'zbek aralash)
- ✅ Long-form audio (30 soniyadan ortiq)

**Kamchiliklari:**

- ❌ Eng katta model (6GB - Large)
- ❌ Sekin inference (real-time emas)
- ❌ Hallucinations (ba'zan keraksiz so'z qo'shadi)

## 📈 Model Taqqoslash Strategiyasi

### Evaluation Metrikalari

#### 1. WER (Word Error Rate)

```
WER = (S + D + I) / N × 100%
S = Substitutions (noto'g'ri so'z)
D = Deletions (tushib qolgan so'z)
I = Insertions (qo'shilgan so'z)
N = Reference so'zlar soni
```

**Misol:**

- Reference: "Men Toshkentda yashayman"
- Hypothesis: "Men Toshkenda yashayman"
- WER = 1/3 × 100% = 33.3% (1 substitution)

#### 2. CER (Character Error Rate)

```
CER = (S + D + I) / N × 100%
```

- So'z o'rniga harf bo'yicha hisoblash
- O'zbek tili uchun CER muhim (agglutinativ til)

#### 3. Real-time Factor (RTF)

```
RTF = Processing Time / Audio Duration
```

- RTF < 1.0 = real-time da ishlaydi
- RTF = 0.5 = 10 soniya audio 5 soniyada processing

### Taqqoslash Jadvali (Kutilayotgan Natijalar)

| Model        | WER (Clean) | WER (Noisy) | RTF | Model Size | Training Time |
| ------------ | ----------- | ----------- | --- | ---------- | ------------- |
| MFCC-CNN-CTC | 25-35%      | 40-50%      | 0.3 | 5 MB       | 4-6 soat      |
| Wav2Vec2     | 15-20%      | 25-30%      | 0.8 | 1.2 GB     | 2-3 soat      |
| Whisper      | 10-15%      | 15-20%      | 1.5 | 6 GB       | 1-2 soat      |

**Xulosalar:**

- Whisper eng yaxshi accuracy, lekin eng sekin
- MFCC-CNN eng tez, lekin eng past accuracy
- Wav2Vec2 optimal balance (speed vs accuracy)

## 🔊 Shovqinga Qarshi Texnikalar

### 1. Data Augmentation

```python
# Speed perturbation
librosa.effects.time_stretch(y, rate=0.9)  # 10% sekinroq
librosa.effects.time_stretch(y, rate=1.1)  # 10% tezroq

# Pitch shifting
librosa.effects.pitch_shift(y, sr, n_steps=2)  # 2 yarim ton yuqori

# Background noise
noisy_audio = clean_audio + 0.01 * noise
```

### 2. SpecAugment

- Frequency masking - ba'zi chastotalarni mask qilish
- Time masking - ba'zi vaqt oralig'ini mask qilish
- Ba'zi phoneme-lar yo'qolsa ham model o'rganadi

### 3. Clean + Noisy Dataset

- 50% clean environment
- 50% noisy environment (ko'cha, oshxona, avtomobil)
- Model ikkala sharoitda ishlashni o'rganadi

## 🚀 Real-time ASR Tizimi

### Streamlit Web Interface

```python
import streamlit as st
from transformers import pipeline

# Model yuklash
asr_pipeline = pipeline("automatic-speech-recognition",
                        model="./uzbek_wav2vec2")

# Audio kirish
audio = st.audio_input("Gapiring")

# Real-time transcription
if audio:
    transcription = asr_pipeline(audio)
    st.write(transcription["text"])
```

### FastAPI Backend

```python
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    result = asr_model(audio.file)
    return {"text": result}
```

## 📚 Adabiyotlar va Ilmiy Asoslar

### Asosiy Maqolalar

1. **Wav2Vec2 (2020)** - Facebook AI

   - Self-supervised learning
   - 100+ til uchun pre-trained models

2. **Whisper (2022)** - OpenAI

   - 680,000 soat audio
   - Multilingual ASR

3. **CTC Loss (2006)** - Alex Graves

   - Alignment-free training
   - Sequence-to-sequence learning

4. **SpecAugment (2019)** - Google Brain
   - Data augmentation for ASR
   - WER 5% yaxshilash

### O'zbek ASR bo'yicha Tadqiqotlar

- **UzASR** - Toshkent Axborot Texnologiyalari Universiteti
- **CommonVoice Uzbek** - Mozilla tomonidan
- **Uzbek Speech Corpus** - minimal dataset

## 📁 Loyiha Tuzilishi

```
uzbek_asr_thesis/
│
├── data/                          # Dataset va yozish interfeysi
│   ├── dataset_creator.py         # Audio preprocessing
│   ├── record_interface.html      # Web-based recording
│   ├── record_server.py            # FastAPI backend
│   ├── manifest.jsonl              # Dataset metadata
│   ├── clean/                      # Tinch muhit audio
│   ├── noisy/                      # Shovqinli audio
│   └── transcripts/                # Matnli transkripsiyalar
│
├── models/                         # Uchta ASR model
│   ├── mfcc_cnn_ctc.py            # Model 1 (classical)
│   ├── wav2vec2_uzbek.py          # Model 2 (pre-trained)
│   └── whisper_uzbek.py           # Model 3 (SOTA)
│
├── notebooks/                      # Jupyter notebook'lar
│   └── (EDA, visualization)
│
├── experiments/                    # Training logs, checkpoints
│   └── (model weights, WER results)
│
├── literature_review/              # Ilmiy manbalar
│   └── LITERATURE_REVIEW_TEMPLATE.md
│
├── real_time_app/                  # Real vaqt ASR
│   └── app.py                      # Streamlit/FastAPI app
│
├── requirements.txt                # Python kutubxonalar
├── README.md                       # Loyiha qisqacha ma'lumot
├── GETTING_STARTED.md              # Boshlash yo'riqnomasi
└── LOYIHA_HAQIDA.md               # Bu fayl (to'liq tavsif)
```

## 🎓 Dissertatsiya Talablari

### Magistrlik Dissertatsiyasi Uchun

- **Hajm:** 80-100 sahifa
- **Boblar:**
  1. Kirish (muammo, maqsad, vazifalar)
  2. Adabiyotlar sharhi (20+ manba)
  3. Metodologiya (3 model tavsifi, matematik dalillar)
  4. Natijalar (WER/CER, jadvallar, grafiklar)
  5. Xulosa va tavsiyalar

### Ilmiy Hissa

1. **O'zbek tili uchun birinchi ochiq dataset** (10 soat)
2. **Uchta SOTA modelning taqqoslanishi**
3. **Shovqinga chidamlilik tahlili**
4. **Lahja va dialekt ta'siri tadqiqoti**

## ⏱️ Ish Rejasi (Timeline)

### 1-2 Hafta: Dataset Yaratish

- [x] Recording interface yaratish ✅
- [ ] 10 soat audio yozish
- [ ] Transcription qilish
- [ ] Validation va cleaning

### 3-4 Hafta: Model Training

- [ ] Google Colab GPU setup
- [ ] MFCC-CNN-CTC o'rgatish
- [ ] Wav2Vec2 fine-tuning
- [ ] Whisper fine-tuning

### 5-6 Hafta: Evaluation

- [ ] WER/CER hisoblash
- [ ] Shovqin testlari
- [ ] Model taqqoslash
- [ ] Grafiklar va jadvallar

### 7-8 Hafta: Dissertatsiya Yozish

- [ ] Adabiyotlar sharhi
- [ ] Metodologiya bo'limi
- [ ] Natijalar tahlili
- [ ] Xulosa va tavsiyalar

## 🔧 Qanday Ishlatish?

### 1. Dataset Yaratish

```bash
# Server ishga tushirish
cd uzbek_asr_thesis
python record_server.py

# Browser'da ochish
# data/record_interface.html
# Mikrofon bilan 10 soat audio yozish
```

### 2. Model Training

```bash
# MFCC-CNN-CTC
python models/mfcc_cnn_ctc.py

# Wav2Vec2
python models/wav2vec2_uzbek.py

# Whisper
python models/whisper_uzbek.py
```

### 3. Real-time ASR

```bash
# Streamlit app
streamlit run real_time_app/app.py

# Browser: http://localhost:8501
```

## 🎯 Kelajak Rejalari

1. **Dataset kengaytirish**

   - 10 soat → 100 soat
   - Ko'proq lahja va dialect

2. **Model yaxshilash**

   - Custom Uzbek language model
   - Punctuation restoration
   - Speaker diarization

3. **Production deployment**

   - Docker containerization
   - FastAPI + Redis
   - Load balancing

4. **Mobile app**
   - Android/iOS app
   - Offline ASR
   - Voice assistant

## 📞 Muammolar va Yechimlar

### 1. Server ishlamayapti

```bash
# Virtual environment activate
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Server ishga tushirish
python record_server.py
```

### 2. Mikrofon ishlamayapti

- Browser settings → Permissions → Microphone ✅
- HTTPS kerak (http://localhost ishlamaydi production'da)

### 3. Model training sekin

- Google Colab yoki Kaggle ishlatish (free GPU)
- Batch size kamaytirish
- Mixed precision training (fp16)

## 📊 Kutilayotgan Natijalar

### Dataset

- ✅ 10 soat o'zbek nutqi
- ✅ 5+ lahja
- ✅ 20+ spiker
- ✅ Clean + Noisy

### Model Performance

- 🎯 WER < 20% (clean)
- 🎯 WER < 30% (noisy)
- 🎯 Real-time inference (RTF < 1.0)

### Ilmiy Natija

- 📝 80-100 sahifa dissertatsiya
- 📊 Kamida 20 ta ilmiy manba
- 📈 Comparative analysis (3 models)
- 🔬 Matematik dalillar (CTC, Attention)

## 🏆 Loyiha Ahamiyati

Bu loyiha:

- ✅ **O'zbek ASR sohasiga ilmiy hissa** qo'shadi
- ✅ **Ochiq manbali dataset** yaratadi
- ✅ **Amaliy qo'llash** imkoniyatlari ko'rsatadi
- ✅ **Xalqaro standartlarga** mos keladi
- ✅ **Kelajakda rivojlantirish** uchun asos

---

**Muallif:** Anvar  
**Sanasi:** 2025  
**Versiya:** 1.0  
**Litsenziya:** MIT (ochiq kod)

**Savollar uchun:** GitHub Issues yoki Email
