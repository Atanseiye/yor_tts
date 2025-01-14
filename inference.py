from data_prep import normalize_text
import torch
from vocoder import mel_to_audio
from training_loop import model
from vocoder import vocoder

def infer(text, model, vocoder):
    """
    Full pipeline for text-to-speech inference.
    """
    # Preprocess text
    text = normalize_text(text)

    # Convert text to tensor (tokenize, convert to indices)
    text_tensor = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)

    # Generate mel-spectrogram
    mel_spec = model(text_tensor)

    # Convert mel-spectrogram to audio
    audio = mel_to_audio(mel_spec)
    return audio

# Example inference
yoruba_text = "Bawo ni, ṣe o wa?"
audio_output = infer(yoruba_text, model, vocoder)
