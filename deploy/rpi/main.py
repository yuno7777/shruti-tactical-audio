"""
Turnkey GTCRN Speech Enhancement CLI for Raspberry Pi
Supported Modes:
  1. 'serial': Ingests live PCM stream from ESP32-S3 over UART/USB, enhances, and plays to DAC.
  2. 'duplex': Direct microphone-to-speaker real-time loopback on the Pi.
  3. 'file': Enhances a WAV file using the streaming 16 ms engine and saves the output.
"""

import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf
from engine import GTCRNEngine

def process_file_mode(engine: GTCRNEngine, input_file: str, output_file: str):
    print(f"Reading input: {input_file}")
    audio, fs = sf.read(input_file, dtype='float32')
    if fs != 16000:
        raise ValueError(f"Sample rate must be 16000 Hz, got {fs}")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    total_chunks = len(audio) // 256
    latencies = []
    enhanced_chunks = []
    late_count = 0

    t_start = time.perf_counter()
    for i in range(total_chunks):
        chunk = audio[i * 256 : (i + 1) * 256]
        enh_chunk, lat, is_late = engine.process_frame(chunk)
        enhanced_chunks.append(enh_chunk)
        latencies.append(lat)
        if is_late:
            late_count += 1

    total_time = time.perf_counter() - t_start
    audio_dur = total_chunks * 0.016
    rtf = total_time / audio_dur

    enhanced_audio = np.concatenate(enhanced_chunks)
    sf.write(output_file, enhanced_audio, 16000)

    print("=======================================================")
    print("                FILE PROCESSING COMPLETE               ")
    print(f"Output File:     {output_file}")
    print(f"Audio Duration:  {audio_dur:.2f} s")
    print(f"Execution Time:  {total_time:.3f} s")
    print(f"Real-Time Factor:{rtf:.4f} ({1.0/rtf:.1f}x real-time)")
    print(f"Average Latency: {np.mean(latencies):.2f} ms")
    print(f"P95 Latency:     {np.percentile(latencies, 95):.2f} ms")
    print(f"Late Frames:     {late_count}/{total_chunks} ({(late_count/total_chunks)*100:.2f}%)")
    print("=======================================================")

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi GTCRN Tactical Speech Enhancer")
    parser.add_argument("--mode", choices=["serial", "duplex", "file"], default="file",
                        help="Operating mode: 'serial' (ESP32-S3), 'duplex' (mic/speaker), 'file' (wav file)")
    parser.add_argument("--model", type=str, default="gtcrn_stream.onnx", help="Path to ONNX model")
    parser.add_argument("--input", type=str, default=None, help="Input WAV file for 'file' mode")
    parser.add_argument("--output", type=str, default="enhanced_output.wav", help="Output WAV file")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port for 'serial' mode")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate for 'serial' mode")
    args = parser.parse_args()

    engine = GTCRNEngine(model_path=args.model)

    if args.mode == "file":
        if not args.input:
            print("Error: --input WAV file must be specified in 'file' mode.")
            sys.exit(1)
        process_file_mode(engine, args.input, args.output)

    elif args.mode == "serial":
        from esp32_receiver import ESP32AudioReceiver
        receiver = ESP32AudioReceiver(port=args.port, baudrate=args.baud, engine=engine)
        receiver.stream_loop()

    elif args.mode == "duplex":
        from alsa_stream import run_duplex
        run_duplex(model_path=args.model)

if __name__ == "__main__":
    main()
