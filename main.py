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
try:
    model = joblib.load("model/voice_ai_detector.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# ------------------------------
# FASTAPI APP
# ------------------------------
app = FastAPI(title="AI Voice Fraud Detection API")

# ------------------------------
# ENABLE CORS
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# API KEY
# ------------------------------
API_KEY = "AB_live_aayush8628"

# ------------------------------
# REQUEST SCHEMA
# ------------------------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ------------------------------
# FEATURE EXTRACTION
# ------------------------------
def extract_features(y, sr):
    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.mean(np.abs(harmonic)) / (
        np.mean(np.abs(percussive)) + 1e-6
    )
    features.append(harmonic_ratio)

    return np.array(features).reshape(1, -1)

# ------------------------------
# MAIN ENDPOINT
# ------------------------------
@app.post("/api/voice-detection")
def detect_voice(data: VoiceRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    allowed_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    if data.language not in allowed_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format supported")

    # Decode Base64
    try:
        audio_bytes = base64.b64decode(data.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # Save temporary file
    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            temp_audio_path = tmp.name

        # Load audio (force mono for stability)
        y, sr = librosa.load(temp_audio_path, sr=None, mono=True)

        if y is None or len(y) == 0:
            raise HTTPException(status_code=400, detail="Empty audio")

        # Chunk detection
        chunk_duration = 3
        chunk_samples = chunk_duration * sr

        total_chunks = 0
        ai_chunks = 0
        ai_probabilities = []

        for start in range(0, len(y), chunk_samples):
            chunk = y[start:start + chunk_samples]

            if len(chunk) < sr:
                continue

            features = extract_features(chunk, sr)

            prediction = model.predict(features)[0]
            prob = model.predict_proba(features)[0]

            ai_probability = prob[1]  # Probability of AI class
            ai_probabilities.append(ai_probability)

            total_chunks += 1

            if prediction == 1:
                ai_chunks += 1

        if total_chunks == 0:
            raise HTTPException(status_code=400, detail="Audio too short")

        ai_percentage = round((ai_chunks / total_chunks) * 100)
        avg_confidence = round(float(np.mean(ai_probabilities)), 2)

        # Final classification
        if ai_percentage > 50:
            label = "AI_GENERATED"
            explanation = "Majority of segments show synthetic voice characteristics."
        elif 0 < ai_percentage <= 50:
            label = "PARTIALLY_AI"
            explanation = "Mixed voice detected with partial AI-generated segments."
        else:
            label = "HUMAN"
            explanation = "Natural pitch and harmonic variations detected."

        return {
            "status": "success",
            "language": data.language,
            "classification": label,
            "confidenceScore": avg_confidence,
            "explanation": explanation
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing failed: {str(e)}")

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
