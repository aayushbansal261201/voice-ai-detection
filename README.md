# 🎙️ AI Voice Fraud Detection API

An AI-powered system that detects whether a given audio clip is **Human-Spoken** or **AI-Generated Speech**.

This solution is designed to prevent **voice-based fraud, impersonation attacks, and AI voice spoofing**, supporting secure audio verification across multiple Indian languages.

---

## 🚀 Problem Overview

With the rapid advancement of generative AI, synthetic voice models can now produce extremely realistic human-like speech. These AI-generated voices are increasingly used in:

- 📞 Phishing calls impersonating family members
- 🏦 Fake bank or customer support calls
- 💳 UPI and financial fraud
- 🧑‍💼 Identity impersonation attacks
- 🏛 Public announcement spoofing

Traditional verification systems cannot reliably detect AI-generated voice.

This system acts as a **defensive AI layer** to identify synthetic voice patterns and prevent fraud.

---

## 🌍 Supported Languages

The system supports detection across the following languages:

- Tamil
- English
- Hindi
- Malayalam
- Telugu

The model is trained and evaluated to generalize across different phonetic structures and speech patterns.

---

## 🧠 How the System Works

### Step 1: Audio Input
The API accepts one MP3 audio file encoded in Base64 format.

### Step 2: Base64 Decoding
The Base64 string is decoded back into binary audio format.

### Step 3: Audio Preprocessing
Using audio signal processing techniques:
- Noise normalization
- Sampling rate standardization
- Feature extraction

### Step 4: Feature Extraction
Extracted features include:
- Mel Spectrogram
- MFCC (Mel Frequency Cepstral Coefficients)
- Spectral patterns
- Temporal frequency characteristics

### Step 5: Deep Learning Classification
A Convolutional Neural Network (CNN) analyzes spectrogram textures and voice artifacts to detect:
- Unnatural pitch consistency
- Robotic tone patterns
- Synthetic waveform smoothness
- Lack of natural human micro-variations

### Step 6: Prediction Output
The model outputs:
- Classification (AI_GENERATED / HUMAN)
- Confidence Score
- Explanation of decision

---

## 🔐 API Architecture

The system is deployed as a secure REST API.

### Endpoint

