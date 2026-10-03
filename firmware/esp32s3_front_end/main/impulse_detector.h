/**
 * @file impulse_detector.h
 * @brief Sub-millisecond Gunshot & High-Energy Transient Limiter for ESP32-S3
 *
 * Designed for 16 kHz 16-bit mono audio streams.
 * Operates on single samples with zero lookahead (62.5 microsecond instantaneous response)
 * to protect downstream neural networks (GTCRN on Raspberry Pi) from ADC saturation
 * and explosive acoustic distortion.
 */

#ifndef IMPULSE_DETECTOR_H
#define IMPULSE_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int16_t threshold_peak;      // Threshold for instantaneous peak detection (default: 16384, ~0.5 FS)
    int16_t threshold_slew;      // Slew rate threshold (sample-to-sample delta)
    float release_coeff;         // Envelope decay factor per sample (e.g., 0.998 for ~30ms release)
    float current_gain;          // Current attenuation gain (0.0 to 1.0)
    int16_t prev_sample;         // Previous sample for slew rate detection
    uint32_t impulse_counter;    // Total impulses detected
    bool in_impulse_state;       // True if currently in active attenuation recovery
} impulse_limiter_t;

/**
 * @brief Initializes the impulse limiter structure with recommended defaults.
 * @param limiter Pointer to impulse_limiter_t
 * @param sample_rate Audio sampling rate in Hz (typically 16000)
 */
void impulse_limiter_init(impulse_limiter_t *limiter, uint32_t sample_rate);

/**
 * @brief Processes an audio buffer in-place with instantaneous transient limiting.
 * @param limiter Pointer to impulse_limiter_t
 * @param pcm_samples Pointer to int16_t audio buffer
 * @param num_samples Number of samples (e.g., 256 for a 16ms frame)
 * @param impulse_flag_out Output pointer set to true if an impulse was triggered in this frame
 */
void impulse_limiter_process(impulse_limiter_t *limiter,
                             int16_t *pcm_samples,
                             size_t num_samples,
                             bool *impulse_flag_out);

#ifdef __cplusplus
}
#endif

#endif // IMPULSE_DETECTOR_H
