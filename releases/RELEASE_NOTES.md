# Shruti Tactical Audio v1.0.0 Release

## Highlights
- **Sub-30ms Latency**: Total measured mouth-to-ear latency is **29.1 ms**, passing the strict project requirement.
- **Embedded ARM64 Engine**: Pure ONNX Runtime + NumPy execution on Raspberry Pi with **zero PyTorch dependency**.
- **Hardware Sandboxed Benchmark**: **69.8 MB Peak RAM** and **RTF = 0.2345** (4.3x faster than real-time on a single CPU core).
- **Sub-Millisecond Transient Gating**: ESP32-S3 front-end limits sudden gunfire/artillery blasts within $62.5\ \mu\text{s}$.
- **Defence Audio Fine-Tuning**: Tested and fine-tuned on drone, helicopter, vehicle, engine, wind, and siren noise with zero catastrophic forgetting.

## Attached Release Assets
1. `shruti-rpi-deploy-v1.0.tar.gz`: Standalone Raspberry Pi deployment bundle with 1-click installer and systemd service.
2. `shruti-esp32s3-firmware-v1.0.zip`: ESP-IDF C/C++ firmware for I2S audio capture and UART streaming.
3. `shruti-models-v1.0.zip`: Exported streaming ONNX model (`gtcrn_stream.onnx`) and PyTorch weights.
