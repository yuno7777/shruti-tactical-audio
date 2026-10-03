"""
Audio Quality Evaluation Script:
Evaluates Clean, Noisy, and Enhanced audio files.
Computes:
- PESQ (Perceptual Evaluation of Speech Quality, Wideband P.862.2)
- STOI (Short-Time Objective Intelligibility)
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) and SI-SDR improvement
- SNR (Signal-to-Noise Ratio) and SNR improvement
"""
import os
import sys
import argparse
import json
import numpy as np
import soundfile as sf

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.metrics import evaluate_audio_quality

def run_evaluation(clean_path: str, noisy_path: str, enhanced_path: str, output_json: str = None) -> dict:
    clean, fs_c = sf.read(clean_path, dtype='float32')
    noisy, fs_n = sf.read(noisy_path, dtype='float32')
    enhanced, fs_e = sf.read(enhanced_path, dtype='float32')

    if not (fs_c == fs_n == fs_e == 16000):
        print(f"Warning: Sample rates must be 16000 Hz. Found clean={fs_c}, noisy={fs_n}, enhanced={fs_e}")

    metrics = evaluate_audio_quality(clean, noisy, enhanced, fs=16000)

    print("\n========================================================")
    print("               AUDIO QUALITY EVALUATION                 ")
    print("========================================================")
    print(f"Clean Audio:    {clean_path}")
    print(f"Noisy Audio:    {noisy_path}")
    print(f"Enhanced Audio: {enhanced_path}")
    print("--------------------------------------------------------")
    print(f"  Metric              |  Noisy   | Enhanced | Improvement")
    print("--------------------------------------------------------")
    print(f"  PESQ (Wideband)     |  {str(metrics['pesq_noisy']):7s} |  {str(metrics['pesq_enhanced']):7s} |  {metrics['pesq_enhanced'] - metrics['pesq_noisy'] if isinstance(metrics['pesq_enhanced'], (int, float)) and isinstance(metrics['pesq_noisy'], (int, float)) else 'N/A'}")
    print(f"  STOI [0 - 1]        |  {str(metrics['stoi_noisy']):7s} |  {str(metrics['stoi_enhanced']):7s} |  {round(metrics['stoi_enhanced'] - metrics['stoi_noisy'], 4) if isinstance(metrics['stoi_enhanced'], (int, float)) and isinstance(metrics['stoi_noisy'], (int, float)) else 'N/A'}")
    print(f"  SI-SDR (dB)         |  {str(metrics['sisdr_noisy_db']):7s} |  {str(metrics['sisdr_enhanced_db']):7s} |  {metrics['sisdr_improvement_db']} dB")
    print(f"  SNR (dB)            |  {str(metrics['snr_noisy_db']):7s} |  {str(metrics['snr_enhanced_db']):7s} |  {metrics['snr_improvement_db']} dB")
    print("========================================================\n")

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation metrics to: {output_json}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio speech enhancement quality")
    parser.add_argument("--clean", type=str, required=True, help="Path to reference clean speech wav")
    parser.add_argument("--noisy", type=str, required=True, help="Path to noisy mixture wav")
    parser.add_argument("--enhanced", type=str, required=True, help="Path to enhanced speech wav")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    run_evaluation(args.clean, args.noisy, args.enhanced, args.out)
