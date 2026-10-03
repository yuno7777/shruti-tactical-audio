"""
Real-Time Duplex Audio Streaming (Microphone In -> GTCRN -> Speaker Out)
Uses sounddevice (PortAudio / ALSA) with minimal buffering (256 samples = 16 ms).
"""

import sys
import time
import numpy as np
import sounddevice as sd
from engine import GTCRNEngine

def run_duplex(model_path: str = "gtcrn_stream.onnx", in_dev=None, out_dev=None):
    engine = GTCRNEngine(model_path=model_path, num_threads=1)
    sample_rate = 16000
    block_size = 256  # 16 ms

    print("==================================================================")
    print("      REAL-TIME DUPLEX AUDIO STREAMING (ALSA / SOUNDDEVICE)       ")
    print(f"Sample Rate: {sample_rate} Hz | Frame: {block_size} samples (16.0 ms)")
    print("==================================================================")

    stats = {
        "frames": 0,
        "latencies": [],
        "late_chunks": 0,
        "t_start": time.time()
    }

    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"Audio buffer status: {status}", file=sys.stderr)

        chunk = indata[:, 0].astype(np.float32)
        enh, lat, is_late = engine.process_frame(chunk)

        stats["frames"] += 1
        stats["latencies"].append(lat)
        if is_late:
            stats["late_chunks"] += 1

        outdata[:, 0] = enh

    print("Starting audio stream. Speak into mic (Press Ctrl+C to stop)...")
    with sd.Stream(
        samplerate=sample_rate,
        blocksize=block_size,
        dtype='float32',
        channels=1,
        device=(in_dev, out_dev),
        callback=audio_callback
    ):
        try:
            while True:
                time.sleep(3.0)
                if stats["latencies"]:
                    recent = stats["latencies"][-180:]
                    avg_lat = np.mean(recent)
                    p95_lat = np.percentile(recent, 95)
                    late_pct = (stats["late_chunks"] / stats["frames"]) * 100.0
                    print(f"Frames: {stats['frames']} | Avg Lat: {avg_lat:.2f}ms | P95: {p95_lat:.2f}ms | Late: {late_pct:.1f}%")
        except KeyboardInterrupt:
            print("\nStopping audio stream...")

    print("Duplex stream ended cleanly.")

if __name__ == "__main__":
    run_duplex()
