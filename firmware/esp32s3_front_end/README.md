# ESP32-S3 Tactical Audio Front-End & Impulse Limiter

Firmware for the ESP32-S3 microcontroller to capture tactical battlefield audio, execute sub-millisecond gunshot/impulse detection and peak limiting, and stream framed audio packets directly to a Raspberry Pi for GTCRN neural enhancement.

---

## 1. System Architecture

```
[INMP441 / SPH0645 MEMS Mic]
        │
        │ I2S DMA (16 kHz, 16-bit Mono, 256 samples / 16 ms)
        ▼
┌─────────────────────────────────────────────────────────────┐
│                       ESP32-S3                              │
│                                                             │
│  Core 0: Audio Capture Task                                 │
│  ├── Sub-Millisecond Impulse Detector (< 0.1 ms latency)    │
│  ├── Soft Saturation Limiter (Clamps gunshots/blasts <=0.5) │
│  └── Packet Protocol Framer (518-byte packets + CRC16)      │
│                                                             │
│  Core 1: High-Speed UART TX Task                            │
│  └── 921,600 Baud Serial Streaming (5.6 ms transfer time)   │
└─────────────────────────────────────────────────────────────┘
        │
        │ UART (GPIO 17 -> RPi Pin 10 / GPIO 15) or USB-C (CDC)
        ▼
[Raspberry Pi CPU: GTCRN Neural Speech Enhancement (<30 ms Total Latency)]
```

---

## 2. Hardware Wiring & Pinout

### Microphone Wiring (INMP441 / SPH0645)

| Mic Pin | ESP32-S3 Pin | Function | Notes |
| :--- | :--- | :--- | :--- |
| **VDD** | **3.3V** | Power | Regulated 3.3V from ESP32-S3 |
| **GND** | **GND** | Ground | Common system ground |
| **SD** | **GPIO 6** | I2S Data In (DIN) | Serial audio data |
| **WS** | **GPIO 5** | I2S Word Select (LRCK) | Frame clock (16 kHz) |
| **SCK** | **GPIO 4** | I2S Bit Clock (BCLK) | Bit clock (512 kHz) |
| **L/R** | **GND** | Channel Select | Tied to GND for Left Channel |

### ESP32-S3 to Raspberry Pi Interconnect

#### Option A: Direct High-Speed UART (Recommended for lowest latency)
| ESP32-S3 Pin | Raspberry Pi Pin | Function |
| :--- | :--- | :--- |
| **GPIO 17 (TX)** | **Pin 10 (GPIO 15 / RXD0)** | Serial In to Pi |
| **GPIO 18 (RX)** | **Pin 8 (GPIO 14 / TXD0)** | Serial Out from Pi |
| **GND** | **Pin 6 (GND)** | Common Ground |

#### Option B: USB-C Cable (Plug and Play)
Connect a USB-C data cable from the ESP32-S3 USB port directly to any Raspberry Pi USB port. The Pi will recognize it as `/dev/ttyACM0` or `/dev/ttyUSB0`.

---

## 3. Packet Protocol Specifications

Each frame contains exactly **256 samples** (16 ms @ 16 kHz):

```
┌───────────┬───────────┬──────────────┬───────────────┬───────────────────────────┬────────────────┐
│ Sync 1    │ Sync 2    │ Seq Number   │ Flags         │ Audio PCM Data            │ Checksum       │
│ (1 byte)  │ (1 byte)  │ (1 byte)     │ (1 byte)      │ (512 bytes = 256 samples) │ (2 bytes CRC16)│
│ 0xAA      │ 0x55      │ 0x00 .. 0xFF │ Bit 0: Limiter│ int16_t, little-endian    │ CCITT-FALSE    │
└───────────┴───────────┴──────────────┴───────────────┴───────────────────────────┴────────────────┘
```

- **Packet Length:** 518 bytes
- **Frame Rate:** 62.5 packets per second
- **Bandwidth Consumption:** 32.4 KB/s
- **UART Capacity at 921,600 baud:** 92.1 KB/s ($2.8\times$ headroom)
- **Transfer Latency:** 5.6 ms

---

## 4. How to Build & Flash

### Prerequisites
Install [ESP-IDF v5.1 or later](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/):

```bash
# Set up IDF environment
. $HOME/esp/esp-idf/export.sh
```

### Build and Flash
```bash
# Navigate to firmware directory
cd firmware/esp32s3_front_end

# Set target to ESP32-S3
idf.py set-target esp32s3

# Build firmware
idf.py build

# Flash to connected ESP32-S3 (replace COMx or /dev/ttyUSB0)
idf.py -p /dev/ttyUSB0 flash monitor
```

---

## 5. Impulse Limiter Specifications

- **Attack Time:** 1 sample ($62.5\ \mu\text{s}$) — instantaneous response without delay buffer.
- **Threshold:** $\pm 16384$ ($0.5$ full scale).
- **Release Time:** $35\text{ ms}$ exponential recovery ($0.9982$ decay per sample).
- **Curve:** Smooth soft-saturation polynomial preventing digital square-wave harmonics.
