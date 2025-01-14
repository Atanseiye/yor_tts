import os
import re
import unicodedata
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from g2p_en import G2p
import matplotlib.pyplot as plt
import inflect

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



###################################
###      Text Preprocessing     ###
###################################
# Normalize Yoruba text
def normalize_text(text):
    """
    Normalize text to handle Yoruba tonal marks and remove unwanted characters.
    """
    text = unicodedata.normalize("NFKD", text)  # Normalize tonal marks
    text = re.sub(r"[^a-zA-Z0-9À-ÖØ-öø-ÿ\s]", "", text)  # Remove non-alphanumeric
    text = text.lower().strip()
    return text


def expand_numbers_and_abbreviations(text):
    """
    Expand numbers and abbreviations in text.
    """
    engine = inflect.engine()

    # Expand numbers
    words = []
    for word in text.split():
        if word.isdigit():
            words.append(engine.number_to_words(word))
        else:
            words.append(word)
    
    # Handle common abbreviations
    expanded_text = " ".join(words)
    expanded_text = expanded_text.replace("Dr.", "Doctor")
    expanded_text = expanded_text.replace("etc.", "et cetera")
    
    return expanded_text.lower()


def text_to_phonemes(text):
    """
    Convert text to phonemes using g2p.
    """
    g2p = G2p()
    phonemes = g2p(text)
    return " ".join(phonemes)



###################################
###     Sound Preprocessing     ###
###################################

# Convert WAV to mel-spectrogram
def wav_to_mel(file_path, sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256):
    """
    Convert audio file to mel-spectrogram.
    """
    # Load audio
    audio, sr = librosa.load(file_path, sr=sample_rate)

    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # Convert to log scale
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec


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


def augment_audio(audio, sample_rate=22050, method="time_stretch", factor=1.2):
    """
    Apply data augmentation to audio.
    Args:
    - method: Augmentation method ('time_stretch', 'pitch_shift', 'noise').
    - factor: Parameter for augmentation (e.g., speed or pitch).
    """
    if method == "time_stretch":
        return librosa.effects.time_stretch(audio, rate=factor)
    elif method == "pitch_shift":
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=factor)
    elif method == "noise":
        noise = np.random.normal(0, 0.01, len(audio))
        return audio + noise
    else:
        raise ValueError("Unsupported augmentation method.")
