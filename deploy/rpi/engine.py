"""
Standalone Raspberry Pi GTCRN Inference Engine
Pure ONNX Runtime + NumPy execution (Zero PyTorch dependency on device).
Designed for ARM64 CPU-only execution with strictly < 100 MB RAM and < 5 ms per chunk.
"""

import os
import time
import numpy as np
import onnxruntime as ort

class GTCRNEngine:
    def __init__(self, model_path: str = "gtcrn_stream.onnx", num_threads: int = 1):
        if not os.path.exists(model_path):
            # Check relative directory if launched from another path
            alt_path = os.path.join(os.path.dirname(__file__), model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model_path = model_path
        self.num_threads = num_threads

        # Configure session options for embedded ARM CPU
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])

        # Initialize internal cache states (fixed shapes for streaming GTCRN)
        self.conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        self.tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        self.inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)

        # STFT parameters (16 kHz, 32 ms window, 16 ms hop)
        self.n_fft = 512
        self.hop_size = 256
        self.win_size = 512
        self.window = (np.hanning(self.win_size) ** 0.5).astype(np.float32)

        # Buffers for streaming STFT and overlap-add synthesis
        self.in_buffer = np.zeros(self.win_size, dtype=np.float32)
        self.ola_buffer = np.zeros(self.win_size, dtype=np.float32)

    def reset_state(self):
        """Clears state caches when starting a new audio stream."""
        self.conv_cache.fill(0)
        self.tra_cache.fill(0)
        self.inter_cache.fill(0)
        self.in_buffer.fill(0)
        self.ola_buffer.fill(0)

    def process_frame(self, chunk_pcm_float: np.ndarray) -> tuple:
        """
        Processes a single 256-sample frame (16 ms @ 16 kHz).
        Args:
            chunk_pcm_float: np.ndarray of shape (256,), float32 in range [-1.0, 1.0]
        Returns:
            enhanced_pcm_float: np.ndarray of shape (256,), float32
            inference_ms: float, time spent in milliseconds
            is_late: bool, True if processing exceeded the 16 ms budget
        """
        t0 = time.perf_counter()

        # 1. Update STFT FIFO buffer
        self.in_buffer[:self.win_size - self.hop_size] = self.in_buffer[self.hop_size:]
        self.in_buffer[self.win_size - self.hop_size:] = chunk_pcm_float

        # 2. Window and FFT
        windowed = self.in_buffer * self.window
        spec = np.fft.rfft(windowed, n=self.n_fft)  # 257 complex frequency bins

        # 3. Format input tensor [1, 257, 1, 2] (Real, Imag)
        mix_input = np.zeros((1, 257, 1, 2), dtype=np.float32)
        mix_input[0, :, 0, 0] = np.real(spec)
        mix_input[0, :, 0, 1] = np.imag(spec)

        # 4. Neural Network Inference with persistent cache states
        out, self.conv_cache, self.tra_cache, self.inter_cache = self.session.run(
            None, {
                'mix': mix_input,
                'conv_cache': self.conv_cache,
                'tra_cache': self.tra_cache,
                'inter_cache': self.inter_cache
            }
        )

        # 5. Inverse FFT and synthesis windowing
        enh_complex = out[0, :, 0, 0] + 1j * out[0, :, 0, 1]
        synth_windowed = np.fft.irfft(enh_complex, n=self.n_fft).astype(np.float32) * self.window

        # 6. Overlap-add
        self.ola_buffer += synth_windowed
        output_chunk = np.copy(self.ola_buffer[:self.hop_size])

        # Shift overlap buffer
        self.ola_buffer[:self.win_size - self.hop_size] = self.ola_buffer[self.hop_size:]
        self.ola_buffer[self.win_size - self.hop_size:] = 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0
        is_late = latency_ms > 16.0

        return output_chunk, latency_ms, is_late
