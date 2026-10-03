/**
 * @file impulse_detector.c
 * @brief Implementation of sub-millisecond impulse limiter for ESP32-S3
 */

#include "impulse_detector.h"
#include <stdlib.h>
#include <math.h>

void impulse_limiter_init(impulse_limiter_t *limiter, uint32_t sample_rate) {
    if (!limiter) return;

    // Peak threshold at ~50% of full scale (16384 out of 32767)
    limiter->threshold_peak = 16384;
    
    // Slew rate threshold (fast rise time typical of gunshots / muzzle blasts)
    limiter->threshold_slew = 12000;

    // Exponential release coefficient for ~35 ms decay envelope
    // release = exp(-1.0 / (sample_rate * 0.035))
    // For 16000 Hz, 16000 * 0.035 = 560 samples. exp(-1/560) ~= 0.9982
    limiter->release_coeff = 0.9982f;

    limiter->current_gain = 1.0f;
    limiter->prev_sample = 0;
    limiter->impulse_counter = 0;
    limiter->in_impulse_state = false;
}

void impulse_limiter_process(impulse_limiter_t *limiter,
                             int16_t *pcm_samples,
                             size_t num_samples,
                             bool *impulse_flag_out) {
    if (!limiter || !pcm_samples) return;

    bool triggered = false;

    for (size_t i = 0; i < num_samples; i++) {
        int16_t s = pcm_samples[i];
        int32_t abs_s = abs((int32_t)s);
        int32_t slew = abs((int32_t)s - (int32_t)limiter->prev_sample);
        limiter->prev_sample = s;

        // Check for impulse conditions
        if (abs_s > limiter->threshold_peak || slew > limiter->threshold_slew) {
            // Compute instantaneous required attenuation
            float target_gain = (float)limiter->threshold_peak / (float)(abs_s > 0 ? abs_s : 1);
            if (target_gain < limiter->current_gain) {
                limiter->current_gain = target_gain;
            }
            triggered = true;
            limiter->in_impulse_state = true;
            limiter->impulse_counter++;
        } else {
            // Smooth release back toward unity gain (1.0)
            limiter->current_gain += (1.0f - limiter->current_gain) * (1.0f - limiter->release_coeff);
            if (limiter->current_gain > 0.999f) {
                limiter->current_gain = 1.0f;
                limiter->in_impulse_state = false;
            }
        }

        // Apply gain attenuation
        float attenuated = (float)s * limiter->current_gain;

        // Soft-saturation polynomial curve to prevent hard clipping harmonics
        // f(x) = x / (1.0 + |x| / 32768.0)
        float normalized = attenuated / 32768.0f;
        float soft_saturated = (normalized / (1.0f + fabsf(normalized) * 0.15f)) * 32768.0f;

        // Clamp to int16 boundaries
        if (soft_saturated > 32767.0f) soft_saturated = 32767.0f;
        if (soft_saturated < -32768.0f) soft_saturated = -32768.0f;

        pcm_samples[i] = (int16_t)soft_saturated;
    }

    if (impulse_flag_out) {
        *impulse_flag_out = triggered;
    }
}
