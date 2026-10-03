import numpy as np
import math

def calculate_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio (SNR) in dB."""
    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return 100.0
    if signal_power < 1e-12:
        return -100.0
    return float(10 * np.log10(signal_power / noise_power))

def calculate_sisdr(clean: np.ndarray, estimated: np.ndarray) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.
    Clean and estimated must be 1D numpy arrays.
    """
    min_len = min(len(clean), len(estimated))
    clean = clean[:min_len]
    estimated = estimated[:min_len]
    clean = clean - np.mean(clean)
    estimated = estimated - np.mean(estimated)
    dot = np.dot(clean, estimated)
    s_target = (dot / (np.dot(clean, clean) + 1e-12)) * clean
    e_noise = estimated - s_target
    target_power = np.sum(s_target ** 2)
    noise_power = np.sum(e_noise ** 2)
    if noise_power < 1e-12:
        return 100.0
    return float(10 * np.log10(target_power / (noise_power + 1e-12)))

def calculate_stoi(clean: np.ndarray, estimated: np.ndarray, fs: int = 16000) -> float:
    """Calculate Short-Time Objective Intelligibility (STOI)."""
    try:
        from pystoi import stoi
        return float(stoi(clean, estimated, fs, extended=False))
    except Exception as e:
        print(f"Warning: STOI calculation failed: {e}")
        return float("nan")

def calculate_pesq(clean: np.ndarray, estimated: np.ndarray, fs: int = 16000) -> float:
    """Calculate Perceptual Evaluation of Speech Quality (PESQ)."""
    try:
        from pesq import pesq
        mode = "wb" if fs == 16000 else "nb"
        return float(pesq(fs, clean, estimated, mode))
    except ImportError:
        return float("nan")
    except Exception as e:
        print(f"Warning: PESQ calculation error: {e}")
        return float("nan")

def evaluate_audio_quality(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray, fs: int = 16000) -> dict:
    """Compute comprehensive audio quality metrics."""
    # Ensure equal length
    min_len = min(len(clean), len(noisy), len(enhanced))
    c = clean[:min_len]
    n = noisy[:min_len]
    e = enhanced[:min_len]

    noisy_snr = calculate_snr(c, n)
    enh_snr = calculate_snr(c, e)
    snr_improvement = enh_snr - noisy_snr

    noisy_sisdr = calculate_sisdr(c, n)
    enh_sisdr = calculate_sisdr(c, e)
    sisdr_improvement = enh_sisdr - noisy_sisdr

    noisy_stoi = calculate_stoi(c, n, fs)
    enh_stoi = calculate_stoi(c, e, fs)

    noisy_pesq = calculate_pesq(c, n, fs)
    enh_pesq = calculate_pesq(c, e, fs)

    return {
        "snr_noisy_db": round(noisy_snr, 2),
        "snr_enhanced_db": round(enh_snr, 2),
        "snr_improvement_db": round(snr_improvement, 2),
        "sisdr_noisy_db": round(noisy_sisdr, 2),
        "sisdr_enhanced_db": round(enh_sisdr, 2),
        "sisdr_improvement_db": round(sisdr_improvement, 2),
        "stoi_noisy": round(noisy_stoi, 4) if not math.isnan(noisy_stoi) else "N/A",
        "stoi_enhanced": round(enh_stoi, 4) if not math.isnan(enh_stoi) else "N/A",
        "pesq_noisy": round(noisy_pesq, 3) if not math.isnan(noisy_pesq) else "N/A",
        "pesq_enhanced": round(enh_pesq, 3) if not math.isnan(enh_pesq) else "N/A"
    }
