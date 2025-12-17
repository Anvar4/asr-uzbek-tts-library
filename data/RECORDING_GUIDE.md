# O'ZBEK NUTQ DATASET YOZISH - QO'LLANMA

## 🎤 Qanday Ishlatish:

### 1-QADAM: Serverni Ishga Tushirish

Terminal ochib, quyidagi buyruqni yozing:

```bash
cd "uzbek_asr_thesis"
python record_server.py
```

Natija:

```
🚀 O'ZBEK ASR RECORDING SERVER
================================================================
📁 Audio papka: C:\Users\anvar\...\data\clean
📝 Transkript papka: C:\Users\anvar\...\data\transcripts
📋 Manifest: C:\Users\anvar\...\data\manifest.jsonl

🌐 Brauzerda oching: data/record_interface.html
================================================================
```

### 2-QADAM: HTML Faylni Ochish

**Brauzerda ochish (2 usul):**

**Usul 1 (Oddiy):**

- `data/record_interface.html` faylni toping
- Fayl ustiga ikki marta bosing
- Chrome/Firefox/Edge da ochiladi

**Usul 2 (VS Code dan):**

- `record_interface.html` ni oching
- O'ng klikka bosing → "Open with Live Server" (agar mavjud bo'lsa)
- yoki faylni brauzerga sudrab tashlang

### 3-QADAM: Audio Yozish

1. **Mikrofon tugmasini bosing** 🎤

   - Birinchi marta ruxsat so'raydi - "Allow" bosing

2. **Gapiring!** 🗣️

   - 3-10 soniya gapiring
   - Tiniq va ravon gapiring
   - Bir jumla yoki qisqa paragraf

3. **To'xtating** ⏹️

   - Mikrofon tugmasini qayta bosing
   - Audio player paydo bo'ladi

4. **Eshitib ko'ring** 👂

   - Play bosib tekshiring
   - Agar yoqmasa - "Qaytadan yozish"

5. **Ma'lumotlarni kiriting** 📝

   - **Transkript**: Nima dedingizni yozing
   - **Spiker ID**: Masalan `sp001` (har bir odam uchun alohida)
   - **Yosh guruhi**: 15-25, 26-35, ...
   - **Jins**: Erkak / Ayol
   - **Lahja**: Toshkent, Farg'ona, ...

6. **Saqlang** 💾

   - "Saqlash" tugmasini bosing
   - Kutib turing...
   - "✅ Muvaffaqiyatli saqlandi" ko'rinadi

7. **Keyingi yozuvga o'ting** 🔄
   - Forma avtomatik tozalanadi
   - Yana boshlang!

---

## 📊 Statistika

Sahifaning yuqori qismida 3 ta raqam:

- **Yozilgan**: Umumiy audio soni
- **Umumiy vaqt**: Barcha audio davomiyligi
- **Bu sessiyada**: Hozir nechta yozdingiz

---

## 🎯 MAQSAD: 10 SOAT AUDIO

**Hisoblash:**

- 1 audio = ~5 soniya
- 1 soat = 720 ta audio (5s \* 720 = 3600s = 1 soat)
- **10 soat = 7200 ta audio** 😱

**Yengilroq yo'l:**

- 10 odam \* 1 soat = 10 soat
- Har kuni 30 daqiqa = 20 kun

---

## ❓ NIMA GAPIRISHNI BILMAYSIZMI?

### Namuna Jumlalar:

**1. Oddiy jumlalar:**

- Salom, mening ismim Anvar
- Bugun havo juda chiroyli
- Men universitetda o'qiyman
- Toshkentda yashayman

**2. Raqamlar:**

- Mening telefon raqamim: nol dokuz bir, uch uch ikki, ...
- Bugun 17-dekabr, 2025-yil
- Manzil: Toshkent shahri, Yunusobod tumani

**3. Savollar:**

- Ishlaringiz qanday?
- Uy vazifangizni qildingizmi?
- Kechagi uchrashuvda qatnashdingizmi?

**4. Kitobdan o'qish:**

- O'zbekiston tarixi kitobidan
- Gazeta yoki yangiliklar
- She'rlar, hikoyalar

**5. Kundalik suhbat:**

- Bugun nima qildingiz?
- Ertaga qanday rejalaringiz bor?
- Sizning sevimli ovqatingiz nima?

---

## 📂 Saqlangan Fayllar

**Audio fayllar:**

```
data/clean/sp001_26-35_m_tashkent_001.wav
data/clean/sp001_26-35_m_tashkent_002.wav
data/clean/sp002_36-45_f_fergana_001.wav
...
```

**Transkriptlar:**

```
data/transcripts/sp001_26-35_m_tashkent_001.txt
data/transcripts/sp001_26-35_m_tashkent_002.txt
...
```

**Manifest (barcha ma'lumotlar):**

```
data/manifest.jsonl
```

---

## ⚠️ MUAMMOLAR VA YECHIMLAR

### ❌ "Server ishlamayapti" xatosi:

**Yechim:** Terminal da `python record_server.py` ishga tushirilganligini tekshiring

### ❌ Mikrofon ishlamayapti:

**Yechim:**

1. Brauzerdagi mikrofon ruxsatini tekshiring
2. Windows Settings → Privacy → Microphone → On

### ❌ Audio saqlanmayapti:

**Yechim:**

1. `data/clean` papkasi mavjudligini tekshiring
2. Terminalda xato xabarlari bormi?

### ❌ Forma to'ldirilmayapti:

**Yechim:** Barcha maydonlar to'ldirilishi kerak (\*)

---

## 🎁 MASLAHATLAR

1. **Sokin xonada yozing** - shovqin kam bo'lsin
2. **Mikrofonga yaqin gapiring** - lekin juda yaqin emas (20-30 cm)
3. **Tiniq va ravon gapiring** - tez emas, asta emas
4. **Har xil jumlalar** - takrorlanmasin
5. **Tanaffus oling** - har 30 daqiqada 5-10 daqiqa
6. **Turli odamlarni jalb qiling** - oila, do'stlar

---

## 📞 YORDAM

Agar biror muammo yuzaga kelsa:

1. Terminalda xato xabarlarini o'qing
2. Browser Console ni oching (F12) - xatolarni ko'ring
3. `data/manifest.jsonl` faylini tekshiring

---

**Omad! 🚀**
