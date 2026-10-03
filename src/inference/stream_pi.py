"""
Raspberry Pi Embedded Streaming Engine for GTCRN Speech Enhancement
Designed specifically for CPU-only ARM64 execution on Raspberry Pi (1 GB RAM, no GPU).
Features:
- Pure ONNX Runtime + NumPy execution (Zero PyTorch runtime dependency on the Pi)
- Fixed 16 ms streaming hop (256 samples @ 16 kHz)
- Real-time late chunk / drop detection (flags when frame processing > 16.0 ms)
- Cache state persistence across chunks
- Hardware thermal monitoring via /sys/class/thermal/thermal_zone0/temp (Linux ARM)
"""
import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf
import onnxruntime as ort

def get_cpu_temperature():
    """Reads CPU temperature on Raspberry Pi Linux, returns None on Windows/x86."""
    temp_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(temp_path):
        try:
            with open(temp_path, "r") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            return None
    return None

class RaspberryPiGTCRNEngine:
    def __init__(self, model_path: str, num_threads: int = 1):
        self.model_path = model_path
        self.num_threads = num_threads

        # Optimize ONNX Runtime for embedded CPU
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])

        # Initialize internal cache states:
        # conv_cache: (2, 1, 16, 16, 33)
        # tra_cache: (2, 3, 1, 1, 16)
        # inter_cache: (2, 1, 33, 16)
        self.conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        self.tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        self.inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)

        # STFT parameters
        self.n_fft = 512
        self.hop_size = 256
        self.win_size = 512
        self.window = (np.hanning(self.win_size) ** 0.5).astype(np.float32)

        # Circular buffer for real-time STFT framing
        self.buffer = np.zeros(self.win_size, dtype=np.float32)
        self.ola_buffer = np.zeros(self.win_size, dtype=np.float32)

    def reset_state(self):
        """Resets streaming caches for a new audio stream."""
        self.conv_cache.fill(0)
        self.tra_cache.fill(0)
        self.inter_cache.fill(0)
        self.buffer.fill(0)
        self.ola_buffer.fill(0)

    def process_frame(self, pcm_chunk_256: np.ndarray) -> tuple:
        """
        Processes 1 streaming chunk (256 samples @ 16 kHz = 16.0 ms).
        Returns (enhanced_pcm_256, latency_ms, is_late).
        """
        t0 = time.perf_counter()

        # 1. Update STFT buffer (FIFO shift by 256 samples)
        self.buffer[:self.win_size - self.hop_size] = self.buffer[self.hop_size:]
        self.buffer[self.win_size - self.hop_size:] = pcm_chunk_256

        # 2. Windowing and RFFT
        windowed = self.buffer * self.window
        spec = np.fft.rfft(windowed, n=self.n_fft)  # 257 complex bins

        # 3. Model input tensor: shape [1, 257, 1, 2]
        mix_input = np.zeros((1, 257, 1, 2), dtype=np.float32)
        mix_input[0, :, 0, 0] = np.real(spec)
        mix_input[0, :, 0, 1] = np.imag(spec)

        # 4. Neural Network Inference
        out, self.conv_cache, self.tra_cache, self.inter_cache = self.session.run(
            [], {
                'mix': mix_input,
                'conv_cache': self.conv_cache,
                'tra_cache': self.tra_cache,
                'inter_cache': self.inter_cache
            }
        )

        # 5. Inverse STFT synthesis
        enh_complex = out[0, :, 0, 0] + 1j * out[0, :, 0, 1]
        synth_windowed = np.fft.irfft(enh_complex, n=self.n_fft).astype(np.float32) * self.window

        # 6. Overlap-add
        self.ola_buffer += synth_windowed
        output_chunk = np.copy(self.ola_buffer[:self.hop_size])

        # Shift OLA buffer
        self.ola_buffer[:self.win_size - self.hop_size] = self.ola_buffer[self.hop_size:]
        self.ola_buffer[self.win_size - self.hop_size:] = 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0
        is_late = latency_ms > 16.0

        return output_chunk, latency_ms, is_late

def run_pi_benchmark(model_path: str, audio_path: str, num_threads: int = 1):
    print("==========================================================================")
    print("             RASPBERRY PI STREAMING BENCHMARK (16 ms CHUNKS)              ")
    print(f"Model: {model_path} | Threads: {num_threads}")
    print("==========================================================================")

    engine = RaspberryPiGTCRNEngine(model_path, num_threads=num_threads)
    audio, fs = sf.read(audio_path, dtype='float32')
    if fs != 16000:
        raise ValueError(f"Audio must be 16000 Hz, got {fs}")

    total_chunks = len(audio) // 256
    latencies = []
    late_chunks = 0
    enhanced_chunks = []

    temp_start = get_cpu_temperature()
    if temp_start is not None:
        print(f"Starting CPU Temperature: {temp_start:.1f} °C")

    t_start = time.perf_counter()
    for i in range(total_chunks):
        chunk = audio[i * 256 : (i + 1) * 256]
        out_chunk, lat, is_late = engine.process_frame(chunk)
        latencies.append(lat)
        enhanced_chunks.append(out_chunk)
        if is_late:
            late_chunks += 1

    total_time_s = time.perf_counter() - t_start
    audio_dur_s = total_chunks * 0.016
    rtf = total_time_s / audio_dur_s

    lat_arr = np.array(latencies)
    avg_lat = float(np.mean(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p95_lat = float(np.percentile(lat_arr, 95))
    max_lat = float(np.max(lat_arr))

    temp_end = get_cpu_temperature()

    print("\n--- Benchmark Performance Results ---")
    print(f"  Processed Audio:     {audio_dur_s:.2f} s ({total_chunks} chunks)")
    print(f"  Total Inference:     {total_time_s:.2f} s")
    print(f"  Real-Time Factor:    {rtf:.4f} (RTF < 1 is real-time)")
    print(f"  Average Latency:     {avg_lat:.2f} ms")
    print(f"  P50 Latency:         {p50_lat:.2f} ms")
    print(f"  P95 Latency:         {p95_lat:.2f} ms")
    print(f"  Max Latency:         {max_lat:.2f} ms")
    print(f"  Late Chunks (>16ms): {late_chunks} / {total_chunks} ({late_chunks/total_chunks*100:.2f}%)")
    if temp_end is not None:
        print(f"  Final CPU Temp:      {temp_end:.1f} °C (Delta: {temp_end - temp_start:+.1f} °C)")
    print("==========================================================================\n")

    # Save output
    enhanced_audio = np.concatenate(enhanced_chunks)
    out_file = "samples/pi_bench_enhanced.wav"
    sf.write(out_file, enhanced_audio, 16000)
    print(f"Saved real-time streaming output to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/fine_tuned/gtcrn_stream.onnx")
    parser.add_argument("--audio", type=str, default="samples/mix.wav")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    run_pi_benchmark(args.model, args.audio, args.threads)
