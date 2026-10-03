"""
Dataset Downloader for Defence Speech Enhancement Pipeline
Downloads specified modular subsets from Hugging Face:
Panav-Payappagoudar/sih-26-processed-audio
"""
import os
import sys
import urllib.request
import zipfile
import argparse

AVAILABLE_SUBSETS = {
    "drone": {
        "file": "Drone-Noise-Audio-set.zip",
        "size_mb": 18.7,
        "description": "UAV / Quadcopter motor whine and aerodynamics"
    },
    "vehicle": {
        "file": "Vehicle-Engine-Wind-Electronic-Electrical-Noise.zip",
        "size_mb": 229.5,
        "description": "Heavy vehicle, engine hums, wind shear, electronic noise"
    },
    "firearms": {
        "file": "firearms-audio-dataset-contains-58-guntypes.zip",
        "size_mb": 20.1,
        "description": "58 distinct gun types / transient acoustic shockwaves for impulse testing"
    },
    "librispeech": {
        "file": "LibriSpeech.zip",
        "size_mb": 969.9,
        "description": "Clean speech recordings standardized at 16kHz mono"
    },
    "metadata": {
        "file": "metadata.csv",
        "size_mb": 17.9,
        "description": "Dataset mapping metadata"
    }
}

BASE_URL = "https://huggingface.co/datasets/Panav-Payappagoudar/sih-26-processed-audio/resolve/main"

def download_file_with_progress(url, dest_path):
    print(f"Downloading from {url} to {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        total_length = response.info().get('Content-Length')
        if total_length is None:
            out_file.write(response.read())
        else:
            dl = 0
            total_length = int(total_length)
            block_size = 65536
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                dl += len(buffer)
                out_file.write(buffer)
                percent = dl / total_length * 100
                sys.stdout.write(f"\rProgress: {percent:.1f}% ({dl / (1024*1024):.1f}/{total_length / (1024*1024):.1f} MB)")
                sys.stdout.flush()
            print()

def download_subset(subset_key, target_dir="data/raw", extract=True):
    if subset_key not in AVAILABLE_SUBSETS:
        raise ValueError(f"Unknown subset: {subset_key}. Available: {list(AVAILABLE_SUBSETS.keys())}")
    
    os.makedirs(target_dir, exist_ok=True)
    info = AVAILABLE_SUBSETS[subset_key]
    filename = info["file"]
    dest_path = os.path.join(target_dir, filename)
    url = f"{BASE_URL}/{filename}"

    print(f"\n[DOWNLOAD] Subset '{subset_key}': {info['description']} (~{info['size_mb']} MB)")
    if not os.path.exists(dest_path):
        download_file_with_progress(url, dest_path)
    else:
        print(f"File {dest_path} already exists. Skipping download.")

    if extract and filename.endswith(".zip"):
        extract_dir = os.path.join(target_dir, filename.replace(".zip", ""))
        print(f"Extracting {dest_path} to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(dest_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extraction complete: {extract_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download subsets of SIH-26 audio dataset")
    parser.add_argument("--subsets", nargs="+", default=["drone", "vehicle", "firearms"],
                        help=f"Subsets to download: {list(AVAILABLE_SUBSETS.keys())}")
    parser.add_argument("--dest", type=str, default="data/raw", help="Target raw data directory")
    parser.add_argument("--no-extract", action="store_true", help="Do not extract downloaded zips")
    args = parser.parse_args()

    for s in args.subsets:
        download_subset(s, target_dir=args.dest, extract=not args.no_extract)
