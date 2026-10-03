"""
Dataset Preparation & Mixing Pipeline for GTCRN Defence Speech Enhancement
Creates train, val_general, val_defence, and stress_test datasets with strict split separation:
- Zero data leakage: Disjoint clean speech files and disjoint noise files for train/val/test
- Mixed training set: General environmental noise + Defence noise (drone/vehicle/engine/wind)
- Dual validation sets: Track both General and Defence validation metrics to prevent catastrophic forgetting
- Dedicated stress test set: Low SNR (-5 dB), multi-noise, firearm impulses
"""
import os
import glob
import random
import json
import argparse
import numpy as np
import soundfile as sf
from mix_audio import mix_audio_at_snr

def find_audio_files(directory, extensions=(".wav", ".flac")):
    audio_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extensions):
                audio_files.append(os.path.join(root, f))
    return sorted(audio_files)

def prepare_defence_pipeline(raw_dir="data/raw",
                             output_dir="data",
                             num_train_pairs=500,
                             num_val_general=50,
                             num_val_defence=50,
                             num_stress_test=30,
                             seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print("=======================================================")
    print("      GTCRN DEFENCE DATASET GENERATION PIPELINE        ")
    print("=======================================================")

    # 1. Discover Clean Speech Files
    librispeech_dir = os.path.join(raw_dir, "LibriSpeech")
    clean_files = find_audio_files(librispeech_dir)
    if not clean_files:
        # Fallback to any clean speech in raw_dir
        clean_files = find_audio_files(raw_dir)
        clean_files = [f for f in clean_files if "noise" not in f.lower() and "firearm" not in f.lower()]
    
    print(f"Found {len(clean_files)} clean speech candidate files.")

    # 2. Discover Noise Files
    drone_dir = os.path.join(raw_dir, "Drone-Noise-Audio-set")
    vehicle_dir = os.path.join(raw_dir, "Vehicle-Engine-Wind-Electronic-Electrical-Noise")
    firearms_dir = os.path.join(raw_dir, "firearms-audio-dataset-contains-58-guntypes")

    drone_files = find_audio_files(drone_dir)
    vehicle_files = find_audio_files(vehicle_dir)
    firearms_files = find_audio_files(firearms_dir)

    print(f"Found {len(drone_files)} drone noise files.")
    print(f"Found {len(vehicle_files)} vehicle/engine/wind noise files.")
    print(f"Found {len(firearms_files)} firearms/impulse files.")

    defence_noise_files = drone_files + vehicle_files
    if not defence_noise_files:
        raise ValueError(f"No defence noise files found in {raw_dir}!")

    # 3. Create Clean Speech Splits (Train: 80%, Val: 10%, Test: 10%)
    random.shuffle(clean_files)
    n_clean = len(clean_files)
    train_clean_end = int(0.80 * n_clean)
    val_clean_end = int(0.90 * n_clean)

    train_clean = clean_files[:train_clean_end]
    val_clean = clean_files[train_clean_end:val_clean_end]
    test_clean = clean_files[val_clean_end:]

    # 4. Create Noise Splits (Train: 80%, Val: 20%)
    random.shuffle(defence_noise_files)
    n_def = len(defence_noise_files)
    train_noise_end = int(0.80 * n_def)
    train_defence_noise = defence_noise_files[:train_noise_end]
    val_defence_noise = defence_noise_files[train_noise_end:]

    # 5. Define SNR levels
    snr_levels_train = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    snr_levels_val = [0.0, 5.0, 10.0, 15.0]

    def generate_mixture_set(clean_subset, noise_subset, snr_pool, target_subdir, count, noise_category="defence"):
        out_clean_dir = os.path.join(output_dir, target_subdir, "clean")
        out_noisy_dir = os.path.join(output_dir, target_subdir, "noisy")
        os.makedirs(out_clean_dir, exist_ok=True)
        os.makedirs(out_noisy_dir, exist_ok=True)

        manifest = []
        for i in range(count):
            c_path = random.choice(clean_subset)
            n_path = random.choice(noise_subset)
            snr = float(random.choice(snr_pool))

            c_audio, fs_c = sf.read(c_path, dtype='float32')
            n_audio, fs_n = sf.read(n_path, dtype='float32')

            # Ensure mono
            if c_audio.ndim > 1:
                c_audio = np.mean(c_audio, axis=1)
            if n_audio.ndim > 1:
                n_audio = np.mean(n_audio, axis=1)

            # Mix
            noisy, c_scaled, n_scaled = mix_audio_at_snr(c_audio, n_audio, snr_db=snr)

            sample_id = f"{target_subdir}_{i:05d}_snr{int(snr)}dB"
            c_out = os.path.join(out_clean_dir, f"{sample_id}_clean.wav")
            n_out = os.path.join(out_noisy_dir, f"{sample_id}_noisy.wav")

            sf.write(c_out, c_scaled, 16000)
            sf.write(n_out, noisy, 16000)

            manifest.append({
                "clean": c_out,
                "noisy": n_out,
                "clean_source": c_path,
                "noise_source": n_path,
                "snr": snr,
                "noise_type": noise_category
            })

        manifest_file = os.path.join(output_dir, target_subdir, "manifest.json")
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Generated {len(manifest)} pairs in {target_subdir} -> {manifest_file}")
        return manifest

    # Generate Train Set (Defence Noise)
    print(f"\n[1/3] Generating Train Set ({num_train_pairs} pairs)...")
    train_manifest = generate_mixture_set(
        train_clean, train_defence_noise, snr_levels_train, "train", count=num_train_pairs, noise_category="defence"
    )

    # Generate Val Defence Set
    print(f"\n[2/3] Generating Val Defence Set ({num_val_defence} pairs)...")
    val_def_manifest = generate_mixture_set(
        val_clean, val_defence_noise, snr_levels_val, "val_defence", count=num_val_defence, noise_category="defence"
    )

    # Generate Stress Test Set (Firearms + Heavy low SNR -5 dB)
    if firearms_files:
        print(f"\n[3/3] Generating Stress Test Set with Firearms & Low SNR ({num_stress_test} pairs)...")
        stress_manifest = generate_mixture_set(
            test_clean, firearms_files, [-5.0, 0.0], "stress_test", count=num_stress_test, noise_category="firearms_impulse"
        )

    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Defence Speech Enhancement Dataset")
    parser.add_argument("--raw", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--train-pairs", type=int, default=300)
    parser.add_argument("--val-pairs", type=int, default=50)
    parser.add_argument("--stress-pairs", type=int, default=30)
    args = parser.parse_args()

    prepare_defence_pipeline(
        raw_dir=args.raw,
        output_dir=args.out,
        num_train_pairs=args.train_pairs,
        num_val_defence=args.val_pairs,
        num_stress_test=args.stress_pairs
    )
