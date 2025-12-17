"""
Audio Recording Server
FastAPI server - HTML dan yuborilgan audio fayllarni qabul qiladi
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from pathlib import Path
import json
from datetime import datetime

app = FastAPI(title="Uzbek ASR Recording Server")

# CORS (HTML fayldan kelgan requestlar uchun)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Papkalarni yaratish
BASE_DIR = Path("./data")
CLEAN_DIR = BASE_DIR / "clean"
NOISY_DIR = BASE_DIR / "noisy"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
MANIFEST_FILE = BASE_DIR / "manifest.jsonl"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
NOISY_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload_recording(
    audio: UploadFile = File(...),
    transcript: str = Form(...),
    speaker_id: str = Form(...),
    age_group: str = Form(...),
    gender: str = Form(...),
    dialect: str = Form(...),
    is_noisy: str = Form("false")
):
    """Audio va transkriptni saqlash"""
    
    try:
        # Noisy yoki clean papkani tanlash
        is_noisy_audio = is_noisy.lower() == "true"
        audio_dir = NOISY_DIR if is_noisy_audio else CLEAN_DIR
        audio_type = "noisy" if is_noisy_audio else "clean"
        
        # Fayl nomini yaratish
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = 1
        
        # Agar fayl mavjud bo'lsa, counterga 1 qo'shish
        while True:
            filename = f"{speaker_id}_{age_group}_{gender}_{dialect}_{counter:03d}.wav"
            filepath = audio_dir / filename
            if not filepath.exists():
                break
            counter += 1
        
        # Audio faylni saqlash
        audio_content = await audio.read()
        with open(filepath, "wb") as f:
            f.write(audio_content)
        
        # Transkriptni saqlash
        transcript_file = TRANSCRIPT_DIR / f"{filename.replace('.wav', '.txt')}"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        # Manifest ga qo'shish
        manifest_entry = {
            "audio_filepath": str(filepath),
            "text": transcript,
            "speaker_id": speaker_id,
            "age_group": age_group,
            "gender": gender,
            "dialect": dialect,
            "audio_type": audio_type,
            "timestamp": timestamp
        }
        
        with open(MANIFEST_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
        
        return JSONResponse({
            "status": "success",
            "filename": filename,
            "message": "Audio va transkript saqlandi"
        })
        
    except Exception as e:
        return JSONResponse(
            {"status": "error", "error": str(e)},
            status_code=500
        )


@app.get("/stats")
async def get_statistics():
    """Dataset statistikasi"""
    
    try:
        # Manifest faylni o'qish
        if not MANIFEST_FILE.exists():
            return {
                "total_recordings": 0,
                "total_speakers": 0,
                "dialects": {},
                "age_groups": {},
                "gender": {}
            }
        
        recordings = []
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                recordings.append(json.loads(line))
        
        # Statistika
        speakers = set(r["speaker_id"] for r in recordings)
        dialects = {}
        age_groups = {}
        gender = {}
        
        for r in recordings:
            dialects[r["dialect"]] = dialects.get(r["dialect"], 0) + 1
            age_groups[r["age_group"]] = age_groups.get(r["age_group"], 0) + 1
            gender[r["gender"]] = gender.get(r["gender"], 0) + 1
        
        return {
            "total_recordings": len(recordings),
            "total_speakers": len(speakers),
            "dialects": dialects,
            "age_groups": age_groups,
            "gender": gender
        }
        
    except Exception as e:
        return JSONResponse(
            {"status": "error", "error": str(e)},
            status_code=500
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "O'zbek ASR Recording Server",
        "status": "running",
        "endpoints": {
            "/upload": "POST - Audio yuklash",
            "/stats": "GET - Statistika",
            "/health": "GET - Server holati"
        }
    }


@app.get("/health")
async def health_check():
    """Server holati"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 O'ZBEK ASR RECORDING SERVER")
    print("=" * 60)
    print(f"📁 Clean audio: {CLEAN_DIR.absolute()}")
    print(f"🔊 Noisy audio: {NOISY_DIR.absolute()}")
    print(f"📝 Transkript: {TRANSCRIPT_DIR.absolute()}")
    print(f"📋 Manifest: {MANIFEST_FILE.absolute()}")
    print("\n🌐 Brauzerda oching: data/record_interface.html")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
