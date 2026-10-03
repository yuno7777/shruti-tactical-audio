"""
ESP32-S3 Serial Audio Receiver & Real-Time Processor
Connects via UART (/dev/ttyUSB0 or /dev/ttyAMA0) or USB CDC (/dev/ttyACM0) at 921,600 baud.
Unpacks binary 518-byte frames, verifies CRC16, feeds audio into GTCRNEngine,
and writes enhanced audio to the DAC / sound card.
"""

import sys
import time
import struct
import serial
import numpy as np
from engine import GTCRNEngine

# Packet Protocol Constants
SYNC_1 = 0xAA
SYNC_2 = 0x55
PACKET_SIZE = 518
SAMPLES_PER_FRAME = 256
PAYLOAD_BYTES = 512

def compute_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

class ESP32AudioReceiver:
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 921600, engine: GTCRNEngine = None):
        self.port = port
        self.baudrate = baudrate
        self.engine = engine or GTCRNEngine()
        self.ser = None
        self.running = False

    def connect(self):
        print(f"Connecting to ESP32-S3 on {self.port} at {self.baudrate} baud...")
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=0.1,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        self.ser.reset_input_buffer()
        print("Connected! Waiting for audio frame sync...")

    def read_packet(self):
        """Finds sync bytes and reads one full 518-byte frame."""
        while self.running:
            # Look for 0xAA
            b1 = self.ser.read(1)
            if not b1 or b1[0] != SYNC_1:
                continue

            # Look for 0x55
            b2 = self.ser.read(1)
            if not b2 or b2[0] != SYNC_2:
                continue

            # Found sync header! Read remaining 516 bytes
            rest = self.ser.read(PACKET_SIZE - 2)
            if len(rest) != PACKET_SIZE - 2:
                continue

            packet_data = b1 + b2 + rest

            # Check CRC16 (last 2 bytes)
            crc_received = struct.unpack("<H", packet_data[-2:])[0]
            crc_calculated = compute_crc16(packet_data[:-2])

            if crc_received != crc_calculated:
                # Checksum error, discard frame
                continue

            seq_id = packet_data[2]
            flags = packet_data[3]
            raw_pcm = np.frombuffer(packet_data[4:516], dtype=np.int16)

            return seq_id, flags, raw_pcm

        return None, None, None

    def stream_loop(self, output_callback=None):
        """Runs the continuous ingestion and enhancement loop."""
        self.running = True
        self.connect()

        frame_count = 0
        impulse_count = 0
        latencies = []
        late_count = 0

        t_last_log = time.time()

        try:
            while self.running:
                seq_id, flags, raw_pcm = self.read_packet()
                if raw_pcm is None:
                    break

                frame_count += 1
                if flags & 0x01:
                    impulse_count += 1

                # Normalize int16 -> float32 [-1.0, 1.0]
                chunk_float = raw_pcm.astype(np.float32) / 32768.0

                # Neural Speech Enhancement
                enh_float, latency_ms, is_late = self.engine.process_frame(chunk_float)
                latencies.append(latency_ms)
                if is_late:
                    late_count += 1

                # Convert back to int16 for audio playback
                enh_int16 = np.clip(enh_float * 32767.0, -32768, 32767).astype(np.int16)

                if output_callback:
                    output_callback(enh_int16)

                # Periodic stats logging every 3 seconds
                if time.time() - t_last_log >= 3.0:
                    avg_lat = np.mean(latencies[-180:]) if latencies else 0.0
                    p95_lat = np.percentile(latencies[-180:], 95) if latencies else 0.0
                    late_pct = (late_count / frame_count) * 100.0 if frame_count else 0.0
                    print(f"Frames: {frame_count:05d} | Latency: avg={avg_lat:.2f}ms, p95={p95_lat:.2f}ms | Late: {late_pct:.1f}% | Impulses: {impulse_count}")
                    t_last_log = time.time()

        except KeyboardInterrupt:
            print("\nStopping audio stream...")
        finally:
            self.running = False
            if self.ser:
                self.ser.close()
            print("Stream closed safely.")

if __name__ == "__main__":
    port_name = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
    receiver = ESP32AudioReceiver(port=port_name)
    receiver.stream_loop()
