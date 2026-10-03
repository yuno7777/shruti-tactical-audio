import soundfile as sf
import numpy as np

def inspect_wav(path):
    data, fs = sf.read(path)
    dur = len(data) / fs
    print(f"File: {path}")
    print(f"  Sample Rate: {fs} Hz")
    print(f"  Shape: {data.shape}")
    print(f"  Duration: {dur:.2f} s")
    print(f"  Max Amplitude: {np.max(np.abs(data)):.4f}")
    print(f"  RMS Energy: {np.sqrt(np.mean(data**2)):.4f}")

if __name__ == "__main__":
    inspect_wav("gtcrn_repo/test_wavs/mix.wav")
