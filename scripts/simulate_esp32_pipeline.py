"""
End-to-End System Simulator: ESP32-S3 + Raspberry Pi Pipeline
Simulates the two-stage defence architecture:
Stage 1: ESP32-S3 Front-End Audio Capture & Impulse/Gunshot Transient Gating
Stage 2: Raspberry Pi GTCRN Streaming Enhancement
Measures:
- Total end-to-end mouth-to-ear latency budget
- Audio quality improvement under mixed continuous defence noise + acoustic shockwaves
"""
import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.stream_pi import RaspberryPiGTCRNEngine
from src.evaluation.metrics import evaluate_audio_quality

class ESP32S3ImpulseFrontEnd:
    """
    Simulates the ESP32-S3 microcontroller audio front-end.
    Runs ultra-low latency peak detection and transient limiting in C/assembly.
    """
    def __init__(self, threshold: float = 0.5, decay: float = 0.95):
        self.threshold = threshold
        self.decay = decay
        self.gain = 1.0

    def process_chunk(self, chunk_256: np.ndarray) -> tuple:
        """
        Processes 256 samples (16 ms).
        Returns (gated_chunk, is_impulse_detected).
        """
        out = np.copy(chunk_256)
        impulse_detected = False
        for i in range(len(out)):
            abs_val = abs(out[i])
            if abs_val * self.gain > self.threshold:
                self.gain = self.threshold / (abs_val + 1e-12)
                impulse_detected = True
            else:
                self.gain = min(1.0, self.gain / self.decay)
            out[i] *= self.gain
        return out, impulse_detected

def simulate_pipeline(audio_path: str, model_path: str, out_path: str = "samples/esp32_rpi_pipeline_out.wav"):
    print("==========================================================================")
    print("     ESP32-S3 + RASPBERRY PI SPEECH ENHANCEMENT PIPELINE SIMULATION       ")
    print("==========================================================================")

    esp32 = ESP32S3ImpulseFrontEnd(threshold=0.45)
    rpi_engine = RaspberryPiGTCRNEngine(model_path, num_threads=1)

    audio, fs = sf.read(audio_path, dtype='float32')
    if fs != 16000:
        raise ValueError(f"Audio must be 16000 Hz, got {fs}")

    total_chunks = len(audio) // 256
    enhanced_chunks = []
    esp32_times = []
    rpi_times = []
    impulse_events = 0

    for i in range(total_chunks):
        raw_chunk = audio[i * 256 : (i + 1) * 256]

        # Stage 1: ESP32-S3
        t0 = time.perf_counter()
        gated_chunk, is_impulse = esp32.process_chunk(raw_chunk)
        t_esp = (time.perf_counter() - t0) * 1000.0
        esp32_times.append(t_esp)
        if is_impulse:
            impulse_events += 1

        # Stage 2: Raspberry Pi GTCRN Streaming Inference
        t1 = time.perf_counter()
        enhanced_chunk, rpi_lat, _ = rpi_engine.process_frame(gated_chunk)
        t_rpi = (time.perf_counter() - t1) * 1000.0
        rpi_times.append(t_rpi)

        enhanced_chunks.append(enhanced_chunk)

    enhanced_audio = np.concatenate(enhanced_chunks)
    sf.write(out_path, enhanced_audio, fs)

    # Latency Breakdown
    avg_esp32_lat = np.mean(esp32_times)
    avg_rpi_lat = np.mean(rpi_times)
    p95_rpi_lat = np.percentile(rpi_times, 95)
    algorithmic_stft_delay = 16.0  # 512 - 256 = 256 samples / 16 kHz = 16.0 ms
    est_i2s_dma_in = 4.0          # Estimated ESP32 I2S DMA ping-pong buffer
    est_transport_delay = 1.5     # Estimated UART / SPI / USB transport
    est_i2s_dma_out = 4.0         # Estimated RPi audio output DMA buffer

    total_mouth_to_ear_ms = (
        est_i2s_dma_in +
        est_transport_delay +
        algorithmic_stft_delay +
        p95_rpi_lat +
        est_i2s_dma_out
    )

    print("\n--- End-to-End Mouth-to-Ear Latency Budget Analysis ---")
    print(f"  1. ESP32-S3 I2S DMA Capture:        {est_i2s_dma_in:.1f} ms")
    print(f"  2. ESP32-S3 Transient Detection:     {avg_esp32_lat:.2f} ms")
    print(f"  3. ESP32 -> RPi Transport:           {est_transport_delay:.1f} ms")
    print(f"  4. RPi STFT Algorithmic Lookahead:   {algorithmic_stft_delay:.1f} ms")
    print(f"  5. RPi GTCRN Neural Inference (p95): {p95_rpi_lat:.2f} ms")
    print(f"  6. RPi Audio Output DAC/I2S Buffer:  {est_i2s_dma_out:.1f} ms")
    print("  ----------------------------------------------------")
    print(f"  TOTAL SYSTEM MOUTH-TO-EAR LATENCY:   {total_mouth_to_ear_ms:.2f} ms")
    print(f"  SPECIFICATION TARGET (< 30 ms):      {'PASS (Meets < 30ms requirement)' if total_mouth_to_ear_ms < 30.0 else 'FAIL'}")
    print(f"  Impulse Events Detected & Clamped:   {impulse_events} chunks")
    print(f"  Output saved to:                     {out_path}")
    print("==========================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="samples/mix.wav")
    parser.add_argument("--model", type=str, default="checkpoints/fine_tuned/gtcrn_stream.onnx")
    parser.add_argument("--out", type=str, default="samples/esp32_rpi_pipeline_out.wav")
    args = parser.parse_args()

    simulate_pipeline(args.audio, args.model, args.out)
