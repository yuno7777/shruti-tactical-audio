"""
Categorized Defence & General Dataset Builder:
Builds structured datasets using ESC-50 metadata, Drone noise, Firearms dataset, and LibriSpeech clean speech.
Enforces:
1. Separation of Defence and General noise categories.
2. Strict train/val disjoint split on noise files and speech files (NO data leakage).
3. Creation of:
   - train: 70% defence noise + 30% general noise across [-5, 0, 5, 10, 15, 20] dB SNR
   - val_defence: unseen defence noise at [0, 5, 10, 15] dB SNR
   - val_general: unseen general noise at [0, 5, 10, 15] dB SNR
   - test_stress: firearms impulses & low SNR (-5 dB)
"""
import os
import csv
import glob
import random
import json
import argparse
import numpy as np
import soundfile as sf
from mix_audio import mix_audio_at_snr

DEFENCE_CATEGORIES = {
    'helicopter', 'airplane', 'engine', 'train',
    'chainsaw', 'car_horn', 'siren', 'wind'
}

GENERAL_CATEGORIES = {
    'rain', 'sea_waves', 'thunderstorm', 'crackling_fire',
    'water_drops', 'insects', 'pouring_water'
}

def load_noise_catalog(raw_dir="data/raw"):
    defence_files = []
    general_files = []

    # 1. ESC-50 catalog
    esc_meta_path = os.path.join(raw_dir, "Vehicle-Engine-Wind-Electronic-Electrical-Noise", "ESC-50-master", "meta", "esc50.csv")
    esc_audio_dir = os.path.join(raw_dir, "Vehicle-Engine-Wind-Electronic-Electrical-Noise", "ESC-50-master", "audio")

    if os.path.exists(esc_meta_path):
        with open(esc_meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = row["category"]
                wav_name = row["filename"]
                wav_path = os.path.join(esc_audio_dir, wav_name)
                if os.path.exists(wav_path):
                    if cat in DEFENCE_CATEGORIES:
                        defence_files.append({"path": wav_path, "category": cat})
                    elif cat in GENERAL_CATEGORIES:
                        general_files.append({"path": wav_path, "category": cat})

    # 2. Drone noise files
    drone_dir = os.path.join(raw_dir, "Drone-Noise-Audio-set")
    drone_wavs = glob.glob(os.path.join(drone_dir, "**", "*.wav"), recursive=True)
    for dw in drone_wavs:
        defence_files.append({"path": dw, "category": "drone_uav"})

    # 3. Firearms files
    firearms_dir = os.path.join(raw_dir, "firearms-audio-dataset-contains-58-guntypes")
    firearms_wavs = glob.glob(os.path.join(firearms_dir, "**", "*.wav"), recursive=True)
    firearms_files = [{"path": fw, "category": "firearms_impulse"} for fw in firearms_wavs]

    print(f"Cataloged {len(defence_files)} defence noise files across categories: {DEFENCE_CATEGORIES | {'drone_uav'}}")
    print(f"Cataloged {len(general_files)} general noise files across categories: {GENERAL_CATEGORIES}")
    print(f"Cataloged {len(firearms_files)} firearms impulse files.")

    return defence_files, general_files, firearms_files

def generate_paired_dataset(clean_files, noise_items, snr_pool, out_dir, count, split_name):
    out_clean = os.path.join(out_dir, split_name, "clean")
    out_noisy = os.path.join(out_dir, split_name, "noisy")
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_noisy, exist_ok=True)

    manifest = []
    for i in range(count):
        clean_item = random.choice(clean_files)
        noise_item = random.choice(noise_items)
        snr = float(random.choice(snr_pool))

        c_audio, fs_c = sf.read(clean_item, dtype='float32')
        n_audio, fs_n = sf.read(noise_item["path"], dtype='float32')

        if c_audio.ndim > 1:
            c_audio = np.mean(c_audio, axis=1)
        if n_audio.ndim > 1:
            n_audio = np.mean(n_audio, axis=1)

        noisy, c_scaled, n_scaled = mix_audio_at_snr(c_audio, n_audio, snr_db=snr)

        file_id = f"{split_name}_{i:05d}_{noise_item['category']}_snr{int(snr)}dB"
        c_path = os.path.join(out_clean, f"{file_id}_clean.wav")
        n_path = os.path.join(out_noisy, f"{file_id}_noisy.wav")

        sf.write(c_path, c_scaled, 16000)
        sf.write(n_path, noisy, 16000)

        manifest.append({
            "clean": c_path,
            "noisy": n_path,
            "clean_source": clean_item,
            "noise_source": noise_item["path"],
            "noise_type": noise_item["category"],
            "snr": snr
        })

    manifest_path = os.path.join(out_dir, split_name, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Created {len(manifest)} pairs in {split_name} -> {manifest_path}")
    return manifest

def build_datasets(raw_dir="data/raw", out_dir="data", n_train=300, n_val_def=40, n_val_gen=40, n_stress=30, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # 1. Clean speech
    librispeech_dir = os.path.join(raw_dir, "LibriSpeech")
    clean_wavs = glob.glob(os.path.join(librispeech_dir, "**", "*.wav"), recursive=True)
    if not clean_wavs:
        clean_wavs = glob.glob(os.path.join(librispeech_dir, "**", "*.flac"), recursive=True)
    
    if not clean_wavs:
        raise FileNotFoundError(f"No clean speech found in {librispeech_dir}. Ensure LibriSpeech is downloaded and extracted.")

    print(f"Total clean speech files: {len(clean_wavs)}")
    random.shuffle(clean_wavs)

    # Split clean speech: 80% train, 10% val, 10% test
    n_c = len(clean_wavs)
    clean_train = clean_wavs[:int(0.80 * n_c)]
    clean_val = clean_wavs[int(0.80 * n_c):int(0.90 * n_c)]
    clean_test = clean_wavs[int(0.90 * n_c):]

    # 2. Noise catalog & split
    def_noise, gen_noise, firearms = load_noise_catalog(raw_dir)
    random.shuffle(def_noise)
    random.shuffle(gen_noise)

    # Noise splits: 80% train, 20% val
    def_train = def_noise[:int(0.80 * len(def_noise))]
    def_val = def_noise[int(0.80 * len(def_noise)):]

    gen_train = gen_noise[:int(0.80 * len(gen_noise))]
    gen_val = gen_noise[int(0.80 * len(gen_noise)):]

    # Mixed training noise: 70% defence + 30% general (Instruction 9: prevent catastrophic forgetting)
    train_noise_pool = def_train + random.sample(gen_train, k=min(len(gen_train), int(0.4 * len(def_train))))

    snrs_train = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    snrs_val = [0.0, 5.0, 10.0, 15.0]

    # Generate sets
    print("\n--- Generating Training Set (Defence + General Mix) ---")
    generate_paired_dataset(clean_train, train_noise_pool, snrs_train, out_dir, n_train, "train")

    print("\n--- Generating Defence Validation Set ---")
    generate_paired_dataset(clean_val, def_val, snrs_val, out_dir, n_val_def, "val_defence")

    print("\n--- Generating General Validation Set ---")
    generate_paired_dataset(clean_val, gen_val, snrs_val, out_dir, n_val_gen, "val_general")

    print("\n--- Generating Stress Test Set (Firearms + Low SNR) ---")
    generate_paired_dataset(clean_test, firearms, [-5.0, 0.0], out_dir, n_stress, "stress_test")

    print("\nAll dataset partitions generated successfully with zero leakage!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--val-def", type=int, default=40)
    parser.add_argument("--val-gen", type=int, default=40)
    parser.add_argument("--stress", type=int, default=30)
    args = parser.parse_args()

    build_datasets(args.raw, args.out, args.train, args.val_def, args.val_gen, args.stress)
