/**
 * @file packet_protocol.h
 * @brief Binary Framing Protocol for ESP32-S3 to Raspberry Pi Audio Streaming
 *
 * Packet Format (518 bytes total):
 * ┌───────────┬───────────┬──────────────┬───────────────┬───────────────────────────┬────────────────┐
 * │ Sync 1    │ Sync 2    │ Seq Number   │ Flags         │ Audio PCM Data            │ Checksum       │
 * │ (1 byte)  │ (1 byte)  │ (1 byte)     │ (1 byte)      │ (512 bytes = 256 samples) │ (2 bytes CRC16)│
 * │ 0xAA      │ 0x55      │ 0x00 .. 0xFF │ Bit 0: Limiter│ int16_t, little-endian    │ CCITT-FALSE    │
 * └───────────┴───────────┴──────────────┴───────────────┴───────────────────────────┴────────────────┘
 *
 * Framing Rate: 16000 Hz / 256 samples = 62.5 packets/sec
 * Bandwidth: 518 bytes * 62.5 = 32,375 bytes/sec (32.4 KB/s)
 * UART Baud: 921,600 baud (capacity: 92,160 bytes/sec, ~2.8x overhead margin)
 */

#ifndef PACKET_PROTOCOL_H
#define PACKET_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#define PACKET_SYNC_BYTE_1       0xAA
#define PACKET_SYNC_BYTE_2       0x55
#define PACKET_SAMPLES_PER_FRAME 256
#define PACKET_PAYLOAD_BYTES     (PACKET_SAMPLES_PER_FRAME * 2) // 512 bytes

#define FLAG_IMPULSE_TRIGGERED   (1 << 0)
#define FLAG_LIMITER_ACTIVE      (1 << 1)

#pragma pack(push, 1)
typedef struct {
    uint8_t  sync1;                                  // 0xAA
    uint8_t  sync2;                                  // 0x55
    uint8_t  sequence_id;                            // Rolling 0..255 counter
    uint8_t  flags;                                  // Bitmask for impulse/limiter events
    int16_t  pcm_samples[PACKET_SAMPLES_PER_FRAME];  // 256 16-bit PCM samples
    uint16_t crc16;                                  // CRC-16-CCITT
} audio_packet_t;
#pragma pack(pop)

/**
 * @brief Computes standard CRC-16-CCITT (polynomial 0x1021, init 0xFFFF).
 */
static inline uint16_t compute_crc16(const uint8_t *data, size_t length) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < length; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    return crc;
}

#endif // PACKET_PROTOCOL_H
