"""
Generate Listening Samples for Fine-Tuned Model:
Enhances real defence test audio (Helicopter, Drone, Vehicle, Mix) and saves paired wavs in samples/
"""
import os
import sys
import glob
import soundfile as sf
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.gtcrn import GTCRN

def generate_samples(model_path="checkpoints/fine_tuned/best_gtcrn_defence.tar",
                     data_dir="data/val_defence",
                     out_dir="samples"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")

    model = GTCRN().to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    print(f"Loaded fine-tuned model: {model_path}")

    # 1. Process mix.wav
    if os.path.exists("samples/mix.wav"):
        mix, fs = sf.read("samples/mix.wav", dtype='float32')
        x_in = torch.view_as_real(
            torch.stft(torch.from_numpy(mix), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=True)
        )
        with torch.no_grad():
            out = model(x_in[None])[0]
        enh = torch.istft(
            torch.view_as_complex(out.contiguous()), 512, 256, 512, torch.hann_window(512).pow(0.5)
        ).numpy()
        sf.write(os.path.join(out_dir, "finetuned_enhanced_mix.wav"), enh, fs)
        print(f"Saved: {out_dir}/finetuned_enhanced_mix.wav")

    # 2. Process sample defence validation files
    noisy_files = sorted(glob.glob(os.path.join(data_dir, "noisy", "*.wav")))[:3]
    for idx, nf in enumerate(noisy_files):
        base_name = os.path.basename(nf).replace("_noisy.wav", "")
        clean_file = os.path.join(data_dir, "clean", f"{base_name}_clean.wav")

        n_audio, fs = sf.read(nf, dtype='float32')
        x_in = torch.view_as_real(
            torch.stft(torch.from_numpy(n_audio), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=True)
        )
        with torch.no_grad():
            out = model(x_in[None])[0]
        enh = torch.istft(
            torch.view_as_complex(out.contiguous()), 512, 256, 512, torch.hann_window(512).pow(0.5)
        ).numpy()

        dest_clean = os.path.join(out_dir, f"sample_{idx}_clean.wav")
        dest_noisy = os.path.join(out_dir, f"sample_{idx}_noisy.wav")
        dest_enh = os.path.join(out_dir, f"sample_{idx}_finetuned_enh.wav")

        if os.path.exists(clean_file):
            c_audio, _ = sf.read(clean_file, dtype='float32')
            sf.write(dest_clean, c_audio, fs)
        sf.write(dest_noisy, n_audio, fs)
        sf.write(dest_enh, enh, fs)
        print(f"Saved sample pair {idx}: {dest_clean}, {dest_noisy}, {dest_enh}")

if __name__ == "__main__":
    generate_samples()
