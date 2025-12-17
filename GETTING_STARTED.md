# QADMA-QADAMLIK YO'RIQNOMA

## O'zbek ASR Dissertatsiyasini Qanday Boshlash

---

## 📋 1-HAFTA: SETUP VA TAYORGARLIK

### Day 1-2: Environment Setup

```bash
# Python environment yaratish
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Kutubxonalar o'rnatish
pip install -r requirements.txt

# GPU tekshirish
python -c "import torch; print(torch.cuda.is_available())"
```

### Day 3-4: Literature Review Boshlash

1. Google Scholar, arXiv, IEEE Xplore dan manbalar topish
2. Har bir manba uchun:
   - PDF yuklab olish
   - Asosiy g'oyalarni yozib olish
   - Citation yozish (APA format)
3. LITERATURE_REVIEW_TEMPLATE.md ni to'ldirish

### Day 5-7: O'zbek Tili O'rganish

- Fonetika
- Morfologiya (agglutinativ struktura)
- Lahjalar
- Mavjud o'zbek ASR ishlarini o'rganish

---

## 📊 2-4 HAFTA: DATASET YARATISH

### Audio Yig'ish Strategiyasi:

**A) Do'stlar va Oiladan:**

- 20-30 kishi
- Har biri 20-30 daqiqa
- Total: ~10 soat

**B) Recording Setup:**

```
Telefon yoki USB mikrofon
Sokin xona
Yaxshi internet (Zoom/Google Meet orqali)
```

**C) Nima o'qitish kerak:**

- Oddiy jumlalar (100-200 ta)
- Kitobdan paragraflar
- Kundalik suhbatlar
- Raqamlar, sanalar, manzillar

### Dataset Structure:

```bash
# Audio fayllarni tartiblash
data/clean/sp001_26-35_m_tashkent_001.wav
           sp001_26-35_m_tashkent_002.wav
           sp002_36-45_f_fergana_001.wav
           ...

# Transkript (har bir audio uchun)
data/transcripts/sp001_26-35_m_tashkent_001.txt
                 sp001_26-35_m_tashkent_002.txt
                 ...
```

### Preprocessing:

```python
python data/dataset_creator.py
```

---

## 🤖 5-10 HAFTA: MODELLARNI TRAINING QILISH

### Model 1: MFCC + CNN + CTC (1 hafta)

```python
# Train script
from models.mfcc_cnn_ctc import MFCC_CNN_CTC, train_one_epoch

model = MFCC_CNN_CTC()
# ... training code
```

**Training tips:**

- Batch size: 16-32
- Learning rate: 1e-4
- Epochs: 50-100
- GPU kerak (Google Colab free tier yetarli)

### Model 2: Wav2Vec2 (2 hafta)

```python
from models.wav2vec2_uzbek import UzbekWav2Vec2Trainer

trainer = UzbekWav2Vec2Trainer(
    model_name="facebook/wav2vec2-large-xlsr-53"
)
trainer.train(train_dataset, eval_dataset)
```

**Tips:**

- Pre-trained model ishlatish (transfer learning)
- Fine-tuning 10-30 epoch
- Learning rate: 1e-4 to 3e-4

### Model 3: Whisper (2 hafta)

```python
from models.whisper_uzbek import UzbekWhisperTrainer

trainer = UzbekWhisperTrainer(model_name="openai/whisper-small")
trainer.train(train_dataset, eval_dataset, num_epochs=10)
```

**Tips:**

- Whisper-small yoki whisper-medium
- Kam epoch kerak (10-15)
- Learning rate: 1e-5

---

## 📈 11-12 HAFTA: EVALUATION VA TAHLIL

### WER/CER Hisoblash:

```python
from jiwer import wer, cer

# Test dataset da baholash
predictions = model.predict(test_dataset)
references = test_dataset['text']

wer_score = wer(references, predictions) * 100
cer_score = cer(references, predictions) * 100

print(f"WER: {wer_score:.2f}%")
print(f"CER: {cer_score:.2f}%")
```

### Shovqinga Bardoshlilik Testi:

```python
# Clean audio
wer_clean = evaluate(model, clean_test)

# Noisy audio (50% SNR)
wer_noisy = evaluate(model, noisy_test)

print(f"WER degradation: {wer_noisy - wer_clean:.2f}%")
```

### Lahjalar bo'yicha Tahlil:

```python
for dialect in ['tashkent', 'fergana', 'khorezm']:
    dialect_data = test_dataset[test_dataset['dialect'] == dialect]
    wer = evaluate(model, dialect_data)
    print(f"{dialect}: WER = {wer:.2f}%")
```

---

## 🚀 13-14 HAFTA: REAL-TIME TIZIM

### Streamlit App:

```bash
streamlit run real_time_app/app.py
```

### FastAPI (Production):

```bash
uvicorn real_time_app:app --reload
```

### Demo Video:

- Mikrofondan yozish
- Fayldan yuklash
- Real-time transcription
- Natijani ko'rsatish

---

## 📝 15-20 HAFTA: DISSERTATSIYA YOZISH

### Dissertatsiya Strukturasi:

**1. KIRISH (5-10 bet)**

- Mavzuning dolzarbligi
- Maqsad va vazifalar
- Ilmiy yangiligi
- Amaliy ahamiyati

**2. ADABIYOTLAR TAHLILI (15-20 bet)**

- Klassik ASR (HMM-GMM)
- Deep Learning ASR (CTC, Attention)
- Wav2Vec2, Whisper
- O'zbek tili tadqiqotlari
- Qiyosiy jadval

**3. NAZARIY QISM (10-15 bet)**

- ASR arxitekturasi
- Feature extraction (MFCC, Mel-spectrogram)
- Neural networks (CNN, LSTM, Transformer)
- CTC Loss
- Attention mechanism
- Matematik isbotlar

**4. METODOLOGIYA (10-15 bet)**

- Dataset yaratish
- Data preprocessing
- Augmentation
- Model arxitekturalari
- Training strategiyasi
- Evaluation metrics

**5. TAJRIBALAR VA NATIJALAR (15-20 bet)**

- 3 ta modelning natijalari
- WER/CER grafiklar
- Shovqinga bardoshlilik
- Lahjalar tahlili
- Qiyoslash (boshqa ishlar bilan)

**6. REAL-TIME TIZIM (5-10 bet)**

- Arxitektura
- Implementation
- Performance
- Demo screenshots

**7. XULOSA (3-5 bet)**

- Natijalar xulosasi
- Ilmiy hissa
- Amaliy ahamiyati
- Kelajak ishlari

**ADABIYOTLAR (20+ manba)**

**ILOVALAR**

- Kod snippets
- Dataset samples
- Full evaluation results

---

## ⚠️ MUHIM MASLAHATLAR

### 1. Vaqt Boshqaruvi:

```
Hafta 1-4:   Literature + Dataset
Hafta 5-10:  Models training
Hafta 11-14: Evaluation + Real-time
Hafta 15-20: Yozish
```

### 2. GPU Maslahat:

- **Google Colab**: Bepul, T4 GPU (15GB)
- **Kaggle**: Bepul, P100 GPU (16GB)
- **Local**: Agar RTX 3060 yoki yuqori bo'lsa

### 3. Dataset Tips:

- Ko'p odamdan oz-oz yig'ish
- Zoom/Google Meet orqali
- Telefon mikrofoni yetarli
- 5-10 sekundlik kliplar optimal

### 4. Model Training Tips:

- Transfer learning ishlatish
- Kichik modeldan boshlash
- Gradient checkpointing (memory tejash)
- Mixed precision training (fp16)

### 5. Dissertation Tips:

- Har hafta 5-10 bet yozish
- Grafiklar va jadvallar ko'proq
- Code snippets qo'shish
- Screenshots (real-time app)

---

## 🎯 HIMOYA UCHUN TAYYORGARLIK

### Javob Berish Kerak Bo'lgan Savollar:

**1. Nima uchun ASR?**
→ Ovozli assistentlar, transkriptsiya, tilni o'rganish...

**2. Nima uchun o'zbek tili?**
→ Low-resource til, dataset yo'q, lahja xilma-xilligi...

**3. Qaysi model eng yaxshi?**
→ Whisper (WER ~X%), lekin Wav2Vec2 tezroq...

**4. Dataset qanday yaratdingiz?**
→ X kishi, Y soat, Z lahja, preprocessing...

**5. WER nima uchun baland/past?**
→ Dataset hajmi, model size, training time...

**6. Real-time qanday ishlaydi?**
→ Streamlit/FastAPI, VAD, chunking, latency <500ms...

**7. Matematik formulalarni tushuntiring:**
→ Bayes (akustik+til model)
→ CTC (alignment-free)
→ Attention (context)

**8. Qanday xatoliklar bor?**
→ Lahja aralashishi, code-switching, shovqin...

**9. Kelajakda nima qilasiz?**
→ Ko'proq data, real-world deployment, dialect-aware...

**10. Amaliy foyda nima?**
→ O'zbek tilida ovozli asistent, call center, accessibility...

---

## 📞 YORDAM KERAK BO'LSA:

1. **Stack Overflow**: PyTorch, Transformers
2. **Hugging Face Forum**: Model issues
3. **GitHub**: Example codes
4. **Reddit**: r/MachineLearning
5. **Discord**: ASR communities

---

**MUVAFFAQIYATLAR! 🎓🚀**

Agar biror qadamda tiqilib qolsangiz, savol bering!
