"""
Real-time ASR Web Application
FastAPI yoki Streamlit yordamida real-time nutqni tanish
"""

import streamlit as st
import torch
import tempfile
import os

# start yani boshlash uchun: python app.py

# ==========================================
# STREAMLIT VERSION (Oddiy va tez)
# ==========================================

def create_streamlit_app():
    """Simple Streamlit app for ASR"""
    
    st.set_page_config(
        page_title="O'zbek ASR Tizimi",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 O'zbek Tili Avtomatik Nutqni Tanish")
    st.markdown("**Whisper / Wav2Vec2 / MFCC-CNN-CTC modellaridan foydalanish**")
    
    # Sidebar - Model selection
    st.sidebar.header("Sozlamalar")
    model_choice = st.sidebar.selectbox(
        "Model tanlash:",
        ["Whisper Small", "Wav2Vec2", "MFCC-CNN-CTC"]
    )
    
    # Load model
    @st.cache_resource
    def load_model(model_name):
        if model_name == "Whisper Small":
            from models.whisper_uzbek import UzbekWhisperTrainer
            return UzbekWhisperTrainer(model_name="./whisper-uzbek")
        elif model_name == "Wav2Vec2":
            from models.wav2vec2_uzbek import UzbekWav2Vec2Trainer
            return UzbekWav2Vec2Trainer(model_name="./wav2vec2-uzbek")
        else:
            # MFCC-CNN-CTC
            from models.mfcc_cnn_ctc import MFCC_CNN_CTC
            model = MFCC_CNN_CTC()
            model.load_state_dict(torch.load("./mfcc_cnn_ctc.pt"))
            return model
    
    try:
        model = load_model(model_choice)
        st.sidebar.success(f"✅ {model_choice} yuklandi")
    except Exception as e:
        st.sidebar.error(f"❌ Model yuklashda xato: {e}")
        model = None
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Audio yuklash yoki yozish")
        
        # Option 1: Upload file
        uploaded_file = st.file_uploader(
            "Audio fayl yuklash (WAV, MP3, FLAC)",
            type=["wav", "mp3", "flac"]
        )
        
        # Option 2: Record audio
        st.markdown("**yoki mikrofondan yozish:**")
        audio_bytes = st.audio_input("Mikrofondan yozish")
        
        if uploaded_file is not None:
            audio_data = uploaded_file
            st.audio(audio_data)
        elif audio_bytes is not None:
            audio_data = audio_bytes
            st.audio(audio_data)
        else:
            audio_data = None
        
        # Transcribe button
        if st.button("🎯 Nutqni tanish", type="primary", disabled=(audio_data is None or model is None)):
            with st.spinner("Nutq tanilmoqda..."):
                try:
                    # Save audio to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_data.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Transcribe
                    if model_choice == "Whisper Small":
                        transcription = model.predict(tmp_path)
                    elif model_choice == "Wav2Vec2":
                        transcription = model.predict(tmp_path)
                    else:
                        # MFCC-CNN-CTC requires more preprocessing
                        transcription = "MFCC-CNN-CTC inference not implemented yet"
                    
                    # Display result
                    st.success("✅ Nutq tanildi!")
                    st.session_state.transcription = transcription
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"❌ Xato: {e}")
    
    with col2:
        st.header("Natija")
        
        if "transcription" in st.session_state:
            st.text_area(
                "Tanilgan matn:",
                value=st.session_state.transcription,
                height=200
            )
            
            # Statistics
            st.metric("So'zlar soni", len(st.session_state.transcription.split()))
            st.metric("Belgilar soni", len(st.session_state.transcription))
            
            # Download button
            st.download_button(
                label="📥 Matnni yuklash",
                data=st.session_state.transcription,
                file_name="transcription.txt",
                mime="text/plain"
            )
        else:
            st.info("Natija bu yerda ko'rsatiladi")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🎓 Magistrlik Dissertatsiyasi - O'zbek Tili ASR Tizimi</p>
            <p>Anvar | 2025</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# FASTAPI VERSION (Real-time WebSocket)
# ==========================================

"""
FastAPI versiyasi:
- Real-time streaming
- WebSocket
- Multiple clients
- Production-ready

Run: uvicorn real_time_app:app --reload
"""

from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse
import asyncio

app = FastAPI(title="Uzbek ASR API")

# Global model
MODEL = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global MODEL
    from models.whisper_uzbek import UzbekWhisperTrainer
    MODEL = UzbekWhisperTrainer(model_name="./whisper-uzbek")
    print("✅ Model loaded")


@app.get("/")
async def get():
    """Landing page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>O'zbek ASR</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
            #transcription { margin-top: 20px; padding: 15px; border: 1px solid #ddd; min-height: 100px; }
        </style>
    </head>
    <body>
        <h1>🎤 O'zbek Tili ASR</h1>
        <input type="file" id="audioFile" accept="audio/*">
        <button onclick="uploadAudio()">Nutqni tanish</button>
        <div id="transcription"></div>
        
        <script>
            async function uploadAudio() {
                const file = document.getElementById('audioFile').files[0];
                if (!file) {
                    alert('Audio faylni tanlang!');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('transcription').innerHTML = 'Kutilmoqda...';
                
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                document.getElementById('transcription').innerHTML = 
                    '<strong>Natija:</strong><br>' + result.transcription;
            }
        </script>
    </body>
    </html>
    """)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio"""
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    # Transcribe
    transcription = MODEL.predict(tmp_path)
    
    # Clean up
    os.unlink(tmp_path)
    
    return {"transcription": transcription}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time streaming ASR
    Client sends audio chunks, server returns transcriptions
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive audio chunk
            audio_bytes = await websocket.receive_bytes()
            
            # Process audio
            # (simplified - real implementation needs buffering)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            # Transcribe
            transcription = MODEL.predict(tmp_path)
            
            # Send result
            await websocket.send_text(transcription)
            
            # Clean up
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# ==========================================
# VOICE ACTIVITY DETECTION (VAD)
# ==========================================

import webrtcvad # pyright: ignore[reportMissingImports]

class VADProcessor:
    """Voice Activity Detection for real-time streaming"""
    
    def __init__(self, aggressiveness=3, sample_rate=16000):
        self.vad = webrtcvad.Vad(aggressiveness)  # 0-3
        self.sample_rate = sample_rate
    
    def is_speech(self, audio_chunk):
        """
        Check if audio chunk contains speech
        
        Args:
            audio_chunk: bytes, must be 10/20/30ms @ 16kHz
        """
        return self.vad.is_speech(audio_chunk, self.sample_rate)
    
    def filter_audio(self, audio_stream):
        """Filter audio stream, keep only speech parts"""
        speech_chunks = []
        
        for chunk in audio_stream:
            if self.is_speech(chunk):
                speech_chunks.append(chunk)
        
        return b''.join(speech_chunks)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    # Streamlit
    create_streamlit_app()
    
    # FastAPI
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
