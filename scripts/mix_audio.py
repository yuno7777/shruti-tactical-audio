"""
Dataset Mixing Pipeline:
Mixes clean speech with additive noise at controlled or randomized SNRs.
Supports:
- Specific SNR levels or random SNR ranges (e.g. [-5, 20] dB)
- Noise looping / random slicing when noise is shorter or longer than speech
- Normalization and clipping prevention
- Exporting metadata json/csv tracking mixture parameters
"""
import os
import random
import numpy as np
import soundfile as sf

def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2) + 1e-12))

def mix_audio_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, target_speech_level_db: float = -26.0) -> tuple:
    """
    Mixes clean speech and noise at a specified SNR (in dB).
    Returns (noisy_speech, clean_scaled, noise_scaled).
    """
    # Align length: if noise is shorter, repeat it; if longer, pick random slice
    if len(noise) < len(clean):
        repeat_times = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeat_times)
    
    if len(noise) > len(clean):
        max_start = len(noise) - len(clean)
        start_idx = random.randint(0, max_start)
        noise = noise[start_idx : start_idx + len(clean)]
    else:
        noise = noise[:len(clean)]

    # Compute energy levels
    clean_rms = compute_rms(clean)
    noise_rms = compute_rms(noise)

    # Normalize clean speech to target level (e.g. -26 dBFS)
    target_rms = 10 ** (target_speech_level_db / 20.0)
    clean_scaled = clean * (target_rms / clean_rms)
    clean_rms = target_rms

    # Calculate required noise RMS for target SNR
    # SNR = 20 * log10(clean_rms / noise_rms)
    # noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    required_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    noise_scaled = noise * (required_noise_rms / (noise_rms + 1e-12))

    noisy = clean_scaled + noise_scaled

    # Prevent clipping if max magnitude exceeds 0.99
    max_val = np.max(np.abs(noisy))
    if max_val > 0.99:
        scale_factor = 0.99 / max_val
        noisy = noisy * scale_factor
        clean_scaled = clean_scaled * scale_factor
        noise_scaled = noise_scaled * scale_factor

    return noisy.astype(np.float32), clean_scaled.astype(np.float32), noise_scaled.astype(np.float32)

if __name__ == "__main__":
    print("mix_audio utility module defined.")
