import joblib
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import librosa
import numpy as np


# 🤖 LOAD TRAINED MODEL

model = joblib.load("model/voice_ai_detector.pkl")


# 🚀 FASTAPI APP

app = FastAPI()

API_KEY = "AB_live_aayush8628"


# 📦 REQUEST BODY SCHEMA

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str



# 🎯 MAIN API ENDPOINT

@app.post("/api/voice-detection")
def detect_voice(
    data: VoiceRequest,
    x_api_key: str = Header(None)
):
    # 🔐 API key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 🌍 Language validation
    allowed_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    if data.language not in allowed_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # 🎧 Audio format validation
    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format supported")

    # 🎵 Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(data.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # 🎵 Save temp MP3 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        temp_audio_path = tmp.name

    # 🎵 Load audio
    try:
        y, sr = librosa.load(temp_audio_path, sr=None)
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read audio")

    if y is None or len(y) == 0:
        raise HTTPException(status_code=400, detail="Empty audio")

    
    # 🎛️ FEATURE EXTRACTION
    

    # 1️⃣ MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc)
    mfcc_std = np.std(mfcc)

    # 2️⃣ Pitch
    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch = pitch[pitch > 0]
    pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0
    pitch_std = np.std(pitch) if len(pitch) > 0 else 0

    # 3️⃣ Energy
    energy = librosa.feature.rms(y=y)[0]
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)

    
    # 🧠 FEATURE VECTOR
    
    features = np.array([
        mfcc_mean,
        mfcc_std,
        pitch_mean,
        pitch_std,
        energy_mean,
        energy_std
    ]).reshape(1, -1)

  
    # 🤖 MODEL PREDICTION
    
    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])

    
    # 🏷️ CLASSIFICATION
    
    label = "AI_GENERATED" if prediction == 1 else "HUMAN"

   
    if prediction == 1:
        explanation = "Unnatural pitch consistency and low energy variation detected"
    else:
        explanation = "Natural pitch and energy variations detected"


    # 📝 RESPONSE
    return {
        "status": "success",
        "language": data.language,
        "classification": label,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
