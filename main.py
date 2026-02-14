import joblib
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import tempfile
import librosa
import numpy as np
import os

# ------------------------------
# LOAD TRAINED MODEL
# ------------------------------
model = joblib.load("model/voice_ai_detector.pkl")

# ------------------------------
# FASTAPI APP
# ------------------------------
app = FastAPI(title="AI Voice Fraud Detection API")

# ------------------------------
# ENABLE CORS (React Support)
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# API KEY
# ------------------------------
API_KEY = "AB_live_aayush8628"

# ------------------------------
# REQUEST BODY SCHEMA
# ------------------------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ------------------------------
# FEATURE EXTRACTION (47 Features)
# ------------------------------
def extract_features(y, sr):
    features = []

    # MFCC (13 mean + 13 std = 26)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    # Harmonic Ratio
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-6)
    features.append(harmonic_ratio)

    return np.array(features).reshape(1, -1)

# ------------------------------
# MAIN API ENDPOINT
# ------------------------------
@app.post("/api/voice-detection")
def detect_voice(data: VoiceRequest, x_api_key: str = Header(None)):

    # API KEY VALIDATION
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Language validation
    allowed_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    if data.language not in allowed_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # Format validation
    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format supported")

    # Decode Base64
    try:
        audio_bytes = base64.b64decode(data.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        temp_audio_path = tmp.name

    # Load audio
    try:
        y, sr = librosa.load(temp_audio_path, sr=None)
    except Exception:
        os.remove(temp_audio_path)
        raise HTTPException(status_code=400, detail="Unable to read audio")

    if y is None or len(y) == 0:
        os.remove(temp_audio_path)
        raise HTTPException(status_code=400, detail="Empty audio")

    # ------------------------------
    # CHUNK-BASED DETECTION
    # ------------------------------
    chunk_duration = 3  # seconds
    chunk_samples = chunk_duration * sr

    total_chunks = 0
    ai_chunks = 0
    confidences = []

    for start in range(0, len(y), chunk_samples):
        end = start + chunk_samples
        chunk = y[start:end]

        if len(chunk) < sr:  # Skip too small chunks
            continue

        features = extract_features(chunk, sr)

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]
        confidence = max(prob)

        total_chunks += 1
        confidences.append(confidence)

        if prediction == 1:
            ai_chunks += 1

    os.remove(temp_audio_path)

    if total_chunks == 0:
        raise HTTPException(status_code=400, detail="Audio too short")

    ai_percentage = round((ai_chunks / total_chunks) * 100)
    avg_confidence = round(float(np.mean(confidences)), 2)

    # FINAL CLASSIFICATION LOGIC
    if ai_percentage > 50:
        label = "AI_GENERATED"
        explanation = "Majority of segments show synthetic voice characteristics."
    elif 0 < ai_percentage <= 50:
        label = "PARTIALLY_AI"
        explanation = "Mixed voice detected with partial AI-generated segments."
    else:
        label = "HUMAN"
        explanation = "Natural pitch and harmonic variations detected."

    # Required format response
    return {
    "status": "success",
    "language": data.language,
    "classification": label,
    "confidenceScore": avg_confidence,
    "explanation": explanation
}

    
