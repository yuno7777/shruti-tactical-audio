import sys, os
sys.path.insert(0, os.path.abspath("."))
import soundfile as sf
from src.evaluation.metrics import evaluate_audio_quality

for i in range(3):
    c_p = f"samples/sample_{i}_clean.wav"
    n_p = f"samples/sample_{i}_noisy.wav"
    e_p = f"samples/sample_{i}_finetuned_enh.wav"
    c, fs = sf.read(c_p)
    n, _ = sf.read(n_p)
    e, _ = sf.read(e_p)
    res = evaluate_audio_quality(c, n, e, fs=fs)
    print(f"=== Sample {i} ===")
    print(f"  SI-SDR: {res['sisdr_noisy_db']} dB -> {res['sisdr_enhanced_db']} dB (Gain: +{res['sisdr_improvement_db']} dB)")
    print(f"  SNR:    {res['snr_noisy_db']} dB -> {res['snr_enhanced_db']} dB (Gain: +{res['snr_improvement_db']} dB)")
    print(f"  STOI:   {res['stoi_noisy']} -> {res['stoi_enhanced']}")
    print(f"  PESQ:   {res['pesq_noisy']} -> {res['pesq_enhanced']}")
