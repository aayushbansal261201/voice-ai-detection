import os
import librosa
import numpy as np
import joblib
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_PATH = "dataset"

X = []
y = []

print("🔄 Loading dataset...")

def extract_features(file_path):
    y_audio, sr = librosa.load(file_path, sr=None)

    features = []

    # -------------------
    # MFCC (13)
    # -------------------
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # -------------------
    # Zero Crossing Rate
    # -------------------
    zcr = librosa.feature.zero_crossing_rate(y_audio)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # -------------------
    # Spectral Centroid
    # -------------------
    centroid = librosa.feature.spectral_centroid(y=y_audio, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # -------------------
    # Spectral Bandwidth
    # -------------------
    bandwidth = librosa.feature.spectral_bandwidth(y=y_audio, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # -------------------
    # Spectral Roll-off
    # -------------------
    rolloff = librosa.feature.spectral_rolloff(y=y_audio, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # -------------------
    # Chroma Features
    # -------------------
    chroma = librosa.feature.chroma_stft(y=y_audio, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    # -------------------
    # Harmonic Ratio
    # -------------------
    harmonic, percussive = librosa.effects.hpss(y_audio)
    harmonic_ratio = np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-6)
    features.append(harmonic_ratio)

    return features

 
# SAFE DATASET LOADING
skipped_files = 0

for label, folder in enumerate(["human", "ai"]):
    folder_path = os.path.join(DATASET_PATH, folder)

    for file in os.listdir(folder_path):
        if file.endswith(".mp3"):
            file_path = os.path.join(folder_path, file)

            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(label)
            except Exception:
                skipped_files += 1
                print(f"⚠ Skipping corrupted file: {file}")

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total valid samples loaded: {len(X)}")
print(f"   Human samples: {sum(y == 0)}")
print(f"   AI samples: {sum(y == 1)}")
print(f"   Skipped corrupted files: {skipped_files}")

# STRATIFIED TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n🚀 Training model...")

start_time = time.time()

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

end_time = time.time()

print(f"⏱ Training completed in {round(end_time - start_time, 2)} seconds")

# EVALUATION
preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)

print("\n📊 MODEL PERFORMANCE")
print("---------------------")
print("Accuracy:", round(accuracy, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["HUMAN", "AI"]))

# SAVE MODEL
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/voice_ai_detector.pkl")

print("\n💾 Model saved successfully as model/voice_ai_detector.pkl")
