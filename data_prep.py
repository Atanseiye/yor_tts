import os
import re
import unicodedata
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Define dataset structure
def prepare_dataset(data_dir, metadata_file):
    """
    Ensure the dataset directory contains:
    - A metadata file (e.g., metadata.csv)
    - Audio files in WAV format.
    """
    metadata_path = os.path.join(data_dir, metadata_file)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata
    metadata = pd.read_csv(metadata_path, sep="|", header=None)
    metadata.columns = ["text", "wav_path", "duration"]
    print(f"Dataset contains {len(metadata)} entries.")
    return metadata



# Normalize Yoruba text
def normalize_text(text):
    """
    Normalize text to handle Yoruba tonal marks and remove unwanted characters.
    """
    text = unicodedata.normalize("NFKD", text)  # Normalize tonal marks
    text = re.sub(r"[^a-zA-Z0-9À-ÖØ-öø-ÿ\s]", "", text)  # Remove non-alphanumeric
    text = text.lower().strip()
    return text


# Convert WAV to mel-spectrogram
def wav_to_mel(wav_path, sample_rate=22050, n_mels=80):
    """
    Convert a WAV file to a mel-spectrogram.
    """
    y, sr = librosa.load(wav_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def trim_silence(audio, top_db=20):
    """
    Trim silence from the beginning and end of the audio.
    Args:
    - audio: Audio signal as a NumPy array.
    - top_db: Decibel threshold for silence.
    Returns:
    - Trimmed audio.
    """
    return librosa.effects.trim(audio, top_db=top_db)[0]


def preprocess_audio(file_path, target_sr=22050):
    """
    Converts audio to a standard format
    Load, resample, normalize, and save audio.
    """
    # Load audio
    audio, sr = librosa.load(file_path, sr=None, mono=True)

    # Resample audio
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Normalize audio
    audio = audio / max(abs(audio))

    # Save preprocessed audio
    sf.write(file_path, audio, target_sr)




# Example usage
data_dir = "./dataset"
metadata_file = "metadata.csv"
metadata = prepare_dataset(data_dir, metadata_file)

# Example usage
wav_path = "./dataset/audio/sample.wav"
mel_spec = wav_to_mel(wav_path)

# Plot mel-spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec, sr=22050, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram')
plt.show()




