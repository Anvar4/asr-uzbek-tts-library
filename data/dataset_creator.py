"""
Dataset yaratish va preprocessing
Audio fayllar yig'ish, transkript qilish, va preprocessing
"""

import os
import json
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd


class DatasetCreator:
    """
    O'zbek nutq dataseti yaratish
    
    Talablar:
    - 10 soat audio
    - 5 yosh guruhi (15-25, 26-35, 36-45, 46-55, 56+)
    - 3 lahja (Toshkent, Farg'ona, Xorazm)
    - Erkak/Ayol balans
    - Toza va shovqinli
    """
    
    def __init__(self, base_dir="./data"):
        self.base_dir = Path(base_dir)
        self.clean_dir = self.base_dir / "clean"
        self.noisy_dir = self.base_dir / "noisy"
        self.transcript_dir = self.base_dir / "transcripts"
        
        # Create directories
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.noisy_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_audio(self, input_path, output_path, target_sr=16000):
        """
        Audio preprocessing:
        - Resample to 16kHz
        - Mono
        - Normalize
        """
        # Load
        audio, sr = librosa.load(input_path, sr=None)
        
        # Resample
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Save
        sf.write(output_path, audio, target_sr)
        
        return len(audio) / target_sr  # duration
    
    def add_noise(self, audio, noise_factor=0.02):
        """Add Gaussian noise"""
        noise = np.random.randn(len(audio))
        noisy_audio = audio + noise_factor * noise
        return noisy_audio
    
    def create_dataset_manifest(self, audio_dir, output_json):
        """
        Dataset manifest yaratish
        
        Format:
        {
            "audio_filepath": "path/to/audio.wav",
            "text": "salom dunyo",
            "duration": 3.5,
            "speaker_id": "speaker_001",
            "age_group": "26-35",
            "gender": "male",
            "dialect": "tashkent"
        }
        """
        manifest = []
        
        # Audio fayllarni scan qilish
        audio_files = list(Path(audio_dir).rglob("*.wav"))
        
        for audio_file in tqdm(audio_files, desc="Creating manifest"):
            # Duration
            audio, sr = librosa.load(audio_file, sr=16000)
            duration = len(audio) / sr
            
            # Transcript (faylning yonidagi txt fayldan o'qish)
            transcript_file = audio_file.with_suffix('.txt')
            if transcript_file.exists():
                text = transcript_file.read_text(encoding='utf-8').strip()
            else:
                text = ""
            
            # Metadata (fayl nomidan chiqariladi)
            # Format: speakerID_ageGroup_gender_dialect_sentenceID.wav
            # Example: sp001_26-35_m_tashkent_001.wav
            filename = audio_file.stem
            parts = filename.split('_')
            
            entry = {
                "audio_filepath": str(audio_file),
                "text": text,
                "duration": duration,
                "speaker_id": parts[0] if len(parts) > 0 else "unknown",
                "age_group": parts[1] if len(parts) > 1 else "unknown",
                "gender": parts[2] if len(parts) > 2 else "unknown",
                "dialect": parts[3] if len(parts) > 3 else "unknown"
            }
            
            manifest.append(entry)
        
        # Save manifest
        with open(output_json, 'w', encoding='utf-8') as f:
            for entry in manifest:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Statistics
        df = pd.DataFrame(manifest)
        total_duration = df['duration'].sum() / 3600  # hours
        
        print(f"✅ Manifest yaratildi: {output_json}")
        print(f"📊 Umumiy fayllar: {len(manifest)}")
        print(f"⏱️  Umumiy davomiylik: {total_duration:.2f} soat")
        print(f"👥 Spikerlar: {df['speaker_id'].nunique()}")
        print(f"📍 Lahjalar: {df['dialect'].unique()}")
        
        return manifest


class AudioDataset:
    """Custom dataset for ASR"""
    
    def __init__(self, manifest_file):
        self.data = []
        
        # Load manifest
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Load audio
        audio, sr = librosa.load(entry['audio_filepath'], sr=16000)
        
        return {
            'audio': audio,
            'text': entry['text'],
            'speaker_id': entry['speaker_id'],
            'duration': entry['duration']
        }


# ==========================================
# DATA AUGMENTATION
# ==========================================

class DataAugmentation:
    """Data augmentation for ASR"""
    
    @staticmethod
    def speed_perturbation(audio, speed_factor):
        """Speed perturbation (0.9x, 1.0x, 1.1x)"""
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    @staticmethod
    def pitch_shift(audio, sr, n_steps):
        """Pitch shifting"""
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def add_noise(audio, noise_factor=0.005):
        """Add Gaussian noise"""
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    
    @staticmethod
    def spec_augment(spectrogram, num_mask=2, freq_masking=10, time_masking=20):
        """
        SpecAugment (Park et al., 2019)
        Mel-spectrogram uchun masking
        """
        spec = spectrogram.copy()
        num_freq, num_time = spec.shape
        
        # Frequency masking
        for _ in range(num_mask):
            f = np.random.randint(0, freq_masking)
            f0 = np.random.randint(0, num_freq - f)
            spec[f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(num_mask):
            t = np.random.randint(0, time_masking)
            t0 = np.random.randint(0, num_time - t)
            spec[:, t0:t0 + t] = 0
        
        return spec


# ==========================================
# TRANSCRIPT QILISH UCHUN HELPER
# ==========================================

def create_transcript_template(audio_files, output_csv):
    """
    Audio fayllar uchun bo'sh transkript CSV yaratish
    
    CSV format:
    filename,text,speaker_id,age_group,gender,dialect
    """
    data = []
    
    for audio_file in audio_files:
        filename = Path(audio_file).name
        data.append({
            'filename': filename,
            'text': '',  # Bo'sh, qo'lda to'ldiriladi
            'speaker_id': '',
            'age_group': '',
            'gender': '',
            'dialect': ''
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"✅ Transkript shabloni yaratildi: {output_csv}")
    print(f"📝 {len(data)} ta fayl uchun transkript yozish kerak")


# ==========================================
# DATASET VALIDATSIYA
# ==========================================

def validate_dataset(manifest_file):
    """
    Dataset validatsiya:
    - Audio fayllar mavjudmi?
    - Transkriptlar to'ldirildimi?
    - Duration to'g'rimi?
    """
    errors = []
    
    with open(manifest_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    for entry in tqdm(data, desc="Validating"):
        # Audio mavjudmi?
        if not Path(entry['audio_filepath']).exists():
            errors.append(f"Audio topilmadi: {entry['audio_filepath']}")
        
        # Transkript bo'shmi?
        if not entry['text']:
            errors.append(f"Transkript bo'sh: {entry['audio_filepath']}")
        
        # Duration musbatmi?
        if entry['duration'] <= 0:
            errors.append(f"Noto'g'ri duration: {entry['audio_filepath']}")
    
    if errors:
        print(f"❌ {len(errors)} ta xato topildi:")
        for error in errors[:10]:  # Birinchi 10 ta
            print(f"  - {error}")
    else:
        print("✅ Dataset validatsiya o'tdi!")
    
    return len(errors) == 0


# ==========================================
# STATISTIKA
# ==========================================

def dataset_statistics(manifest_file):
    """Dataset statistikasi"""
    with open(manifest_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    df = pd.DataFrame(data)
    
    print("=" * 50)
    print("DATASET STATISTIKASI")
    print("=" * 50)
    
    print(f"\n📊 Umumiy:")
    print(f"  - Fayllar soni: {len(df)}")
    print(f"  - Umumiy davomiylik: {df['duration'].sum() / 3600:.2f} soat")
    print(f"  - O'rtacha davomiylik: {df['duration'].mean():.2f} s")
    print(f"  - Min davomiylik: {df['duration'].min():.2f} s")
    print(f"  - Max davomiylik: {df['duration'].max():.2f} s")
    
    print(f"\n👥 Spikerlar:")
    print(f"  - Umumiy spikerlar: {df['speaker_id'].nunique()}")
    print(df['speaker_id'].value_counts().head())
    
    print(f"\n🎂 Yosh guruhlari:")
    print(df['age_group'].value_counts())
    
    print(f"\n⚧️ Jins:")
    print(df['gender'].value_counts())
    
    print(f"\n📍 Lahjalar:")
    print(df['dialect'].value_counts())
    
    print("\n" + "=" * 50)


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    # Dataset yaratish
    creator = DatasetCreator(base_dir="./data")
    
    # 1. Audio fayllarni preprocessing
    # input_audios = list(Path("raw_audio").glob("*.wav"))
    # for audio in tqdm(input_audios):
    #     output = creator.clean_dir / audio.name
    #     creator.preprocess_audio(audio, output)
    
    # 2. Manifest yaratish
    # manifest = creator.create_dataset_manifest(
    #     audio_dir="./data/clean",
    #     output_json="./data/manifest.json"
    # )
    
    # 3. Validatsiya
    # validate_dataset("./data/manifest.json")
    
    # 4. Statistika
    # dataset_statistics("./data/manifest.json")
    
    print("Dataset yaratish to'liq!")
