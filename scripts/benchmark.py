"""
GTCRN CPU-Only Streaming Benchmark
Measures latency (avg, p50, p95, max), RTF, RAM usage, and CPU utilization across chunk sizes.
Strictly CPU-only (no GPU).
"""
import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf
import onnxruntime as ort
import psutil

# Ensure CPU execution only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def run_cpu_streaming_benchmark(onnx_path, audio_path, num_warmup=10, chunk_frames_list=[1, 2, 4]):
    """
    Benchmark ONNX streaming GTCRN on CPU.
    chunk_frames: Number of STFT hops processed per chunk (hop=256 samples @ 16kHz = 16ms per frame).
    """
    print(f"Loading ONNX model: {onnx_path}")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # Standard for embedded single-core or controlled thread tests
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])

    # Read audio
    audio, fs = sf.read(audio_path, dtype='float32')
    if fs != 16000:
        raise ValueError(f"Audio sample rate must be 16000 Hz, got {fs}")
    total_audio_duration_s = len(audio) / fs
    print(f"Loaded audio: {audio_path} ({total_audio_duration_s:.2f}s, {len(audio)} samples)")

    # Prepare STFT parameters
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = np.hanning(win_length) ** 0.5

    # Compute STFT using numpy / scipy
    # Pad audio to match STFT framing
    from scipy.signal import stft
    f, t, Zxx = stft(audio, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft-hop_length, boundary=None)
    # Zxx is shape (F, T) = (257, T)
    spec_real = np.real(Zxx)
    spec_imag = np.imag(Zxx)
    # shape: (1, 257, T, 2)
    spec_input = np.stack([spec_real, spec_imag], axis=-1)[np.newaxis, ...]
    total_frames = spec_input.shape[2]

    process = psutil.Process()
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    results = []

    for chunk_frames in chunk_frames_list:
        chunk_duration_ms = chunk_frames * (hop_length / fs) * 1000.0
        print(f"\n==================================================")
        print(f"Testing Chunk Size: {chunk_frames} frame(s) = {chunk_duration_ms:.1f} ms")
        print(f"==================================================")

        # Initialize caches
        conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)

        # Warmup
        warmup_input = spec_input[:, :, :1, :].astype(np.float32)
        for _ in range(num_warmup):
            session.run([], {
                'mix': warmup_input,
                'conv_cache': conv_cache,
                'tra_cache': tra_cache,
                'inter_cache': inter_cache
            })

        # Reset caches for timed run
        conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)

        latencies_ms = []
        cpu_percents = []

        start_time = time.perf_counter()
        psutil.cpu_percent(interval=None)

        num_steps = total_frames // chunk_frames
        for i in range(num_steps):
            t0 = time.perf_counter()
            for sub_f in range(chunk_frames):
                frame_idx = i * chunk_frames + sub_f
                frame_slice = spec_input[:, :, frame_idx:frame_idx+1, :].astype(np.float32)
                out_i, conv_cache, tra_cache, inter_cache = session.run([], {
                    'mix': frame_slice,
                    'conv_cache': conv_cache,
                    'tra_cache': tra_cache,
                    'inter_cache': inter_cache
                })
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        elapsed_total_s = time.perf_counter() - start_time
        cpu_used = psutil.cpu_percent(interval=None)
        ram_after_mb = process.memory_info().rss / (1024 * 1024)

        audio_processed_s = (num_steps * chunk_frames * hop_length) / fs
        rtf = elapsed_total_s / audio_processed_s

        latencies_arr = np.array(latencies_ms)
        avg_lat = float(np.mean(latencies_arr))
        p50_lat = float(np.percentile(latencies_arr, 50))
        p95_lat = float(np.percentile(latencies_arr, 95))
        max_lat = float(np.max(latencies_arr))
        min_lat = float(np.min(latencies_arr))

        # Algorithmic delay:
        # Window size is 512 samples = 32 ms. Hop size is 256 samples = 16 ms.
        # STFT framing buffer latency = window_length - hop_length = 256 samples = 16 ms algorithmic lookahead/buffering.
        algorithmic_delay_ms = (win_length - hop_length) / fs * 1000.0  # 16.0 ms
        model_p95_latency_ms = p95_lat
        total_model_plus_buffer_p95_ms = algorithmic_delay_ms + model_p95_latency_ms

        res = {
            "chunk_frames": chunk_frames,
            "chunk_duration_ms": chunk_duration_ms,
            "avg_latency_ms": round(avg_lat, 3),
            "p50_latency_ms": round(p50_lat, 3),
            "p95_latency_ms": round(p95_lat, 3),
            "max_latency_ms": round(max_lat, 3),
            "min_latency_ms": round(min_lat, 3),
            "rtf": round(rtf, 4),
            "ram_usage_mb": round(ram_after_mb, 2),
            "cpu_util_percent": round(cpu_used, 1),
            "algorithmic_delay_ms": algorithmic_delay_ms,
            "est_model_plus_buffer_p95_ms": round(total_model_plus_buffer_p95_ms, 3)
        }
        results.append(res)

        print(f"Results for {chunk_duration_ms:.1f}ms chunk:")
        print(f"  Avg Latency: {avg_lat:.2f} ms")
        print(f"  P50 Latency: {p50_lat:.2f} ms")
        print(f"  P95 Latency: {p95_lat:.2f} ms")
        print(f"  Max Latency: {max_lat:.2f} ms")
        print(f"  RTF:         {rtf:.4f} (real-time factor < 1 is real-time)")
        print(f"  RAM Usage:   {ram_after_mb:.1f} MB")
        print(f"  CPU Util:    {cpu_used:.1f} %")
        print(f"  Algorithmic Window Latency: {algorithmic_delay_ms:.1f} ms")
        print(f"  Est. Model+Buffer (p95):    {total_model_plus_buffer_p95_ms:.2f} ms (Target < 30ms)")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-only GTCRN Streaming Benchmark")
    parser.add_argument("--onnx", type=str, default="checkpoints/gtcrn_simple.onnx", help="Path to ONNX model")
    parser.add_argument("--audio", type=str, default="samples/mix.wav", help="Path to input wav")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        # Check alternate locations
        alt = "src/inference/stream/onnx_models/gtcrn_simple.onnx"
        if os.path.exists(alt):
            args.onnx = alt
        else:
            alt2 = "gtcrn_repo/stream/onnx_models/gtcrn_simple.onnx"
            if os.path.exists(alt2):
                args.onnx = alt2

    if not os.path.exists(args.audio):
        alt_audio = "gtcrn_repo/test_wavs/mix.wav"
        if os.path.exists(alt_audio):
            args.audio = alt_audio

    run_cpu_streaming_benchmark(args.onnx, args.audio)
