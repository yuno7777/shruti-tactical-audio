"""
Interactive Real-Time Speech Enhancement Test Bench
Can test:
  1. Live microphone stream (with live ASCII VU meter & latency tracker)
  2. Audio file simulation (streams chunk-by-chunk at exact 16.0 ms real-time pace)
"""

import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deploy", "rpi")))
from engine import GTCRNEngine

def make_vu_meter(level_db: float, width: int = 20) -> str:
    """Generates an ASCII level meter compatible with all Windows/Linux consoles."""
    # level_db from -60 dB to 0 dB
    clamped = max(-60.0, min(0.0, level_db))
    frac = (clamped + 60.0) / 60.0
    filled = int(frac * width)
    return "#" * filled + "-" * (width - filled)

def compute_rms_db(pcm: np.ndarray) -> float:
    rms = np.sqrt(np.mean(pcm ** 2) + 1e-9)
    return 20.0 * np.log10(rms)

def run_file_simulation(engine: GTCRNEngine, wav_path: str, out_path: str):
    print("==========================================================================")
    print("           REAL-TIME CHUNK PACED STREAMING SIMULATION (16 ms)             ")
    print(f"File: {wav_path} -> {out_path}")
    print("==========================================================================")

    audio, fs = sf.read(wav_path, dtype='float32')
    if fs != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {fs} Hz")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    total_chunks = len(audio) // 256
    enhanced_chunks = []
    latencies = []
    late_chunks = 0

    print("\nProcessing frames in real-time (16 ms pace):")
    print("Frame | Noisy Level       | Enhanced Level    | Latency | Status")
    print("----------------------------------------------------------------")

    t_session_start = time.perf_counter()
    for i in range(total_chunks):
        t_frame_start = time.perf_counter()
        chunk = audio[i * 256 : (i + 1) * 256]

        noisy_db = compute_rms_db(chunk)
        enh, lat_ms, is_late = engine.process_frame(chunk)
        enh_db = compute_rms_db(enh)

        enhanced_chunks.append(enh)
        latencies.append(lat_ms)
        if is_late:
            late_chunks += 1

        # Every 10 frames (160 ms), update progress
        if i % 10 == 0 or i == total_chunks - 1:
            noisy_bar = make_vu_meter(noisy_db, width=15)
            enh_bar = make_vu_meter(enh_db, width=15)
            status = "LATE!" if is_late else "OK"
            print(f"\r[{i:04d}/{total_chunks}] [{noisy_bar}] -> [{enh_bar}] | {lat_ms:5.2f}ms | {status}", end="", flush=True)

        # Pace execution at exactly 16.0 ms frame rate
        elapsed = time.perf_counter() - t_frame_start
        if elapsed < 0.016:
            time.sleep(0.016 - elapsed)

    print("\n----------------------------------------------------------------")
    total_time = time.perf_counter() - t_session_start
    audio_dur = total_chunks * 0.016
    enhanced_full = np.concatenate(enhanced_chunks)
    sf.write(out_path, enhanced_full, 16000)

    print("\n=== Streaming Simulation Results ===")
    print(f"Processed Chunks:   {total_chunks} ({audio_dur:.2f}s audio)")
    print(f"Wall Clock Time:    {total_time:.2f}s")
    print(f"Average Model Time: {np.mean(latencies):.2f} ms")
    print(f"P95 Model Time:     {np.percentile(latencies, 95):.2f} ms")
    print(f"Late Chunks:        {late_chunks}/{total_chunks} ({(late_chunks/total_chunks)*100:.2f}%)")
    print(f"Enhanced Audio:     {out_path}")

def run_live_mic(engine: GTCRNEngine):
    import sounddevice as sd
    print("==========================================================================")
    print("           LIVE MICROPHONE STREAM WITH ASCII VU METERS (16 ms)            ")
    print("==========================================================================")
    print("Speak into your microphone. Noise will be suppressed in real-time.")
    print("Press Ctrl+C to exit.\n")
    print("Frame | Noisy Input       | Enhanced Output   | Latency")
    print("---------------------------------------------------------")

    frame_counter = 0

    def callback(indata, outdata, frames, time_info, status):
        nonlocal frame_counter
        chunk = indata[:, 0].astype(np.float32)

        noisy_db = compute_rms_db(chunk)
        enh, lat_ms, is_late = engine.process_frame(chunk)
        enh_db = compute_rms_db(enh)

        outdata[:, 0] = enh

        frame_counter += 1
        if frame_counter % 6 == 0:
            noisy_bar = make_vu_meter(noisy_db, width=15)
            enh_bar = make_vu_meter(enh_db, width=15)
            print(f"\r[{frame_counter:05d}] [{noisy_bar}] -> [{enh_bar}] | {lat_ms:5.2f}ms", end="", flush=True)

    with sd.Stream(samplerate=16000, blocksize=256, channels=1, dtype='float32', callback=callback):
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nExiting live microphone stream.")

def main():
    parser = argparse.ArgumentParser(description="Live GTCRN Speech Enhancement Test Harness")
    parser.add_argument("--mode", choices=["file", "mic"], default="file")
    parser.add_argument("--model", type=str, default="checkpoints/fine_tuned/gtcrn_stream.onnx")
    parser.add_argument("--input", type=str, default="samples/sample_0_noisy.wav")
    parser.add_argument("--output", type=str, default="samples/live_test_output.wav")
    args = parser.parse_args()

    engine = GTCRNEngine(model_path=args.model)

    if args.mode == "file":
        run_file_simulation(engine, args.input, args.output)
    elif args.mode == "mic":
        run_live_mic(engine)

if __name__ == "__main__":
    main()
