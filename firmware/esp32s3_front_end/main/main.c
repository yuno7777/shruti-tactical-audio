/**
 * @file main.c
 * @brief ESP32-S3 Tactical Audio Capture & Sub-Millisecond Impulse Limiter
 *
 * Captures 16 kHz audio via I2S MEMS microphone (INMP441 / SPH0645).
 * Executes instantaneous transient / gunshot detection and peak saturation limiting (<0.1 ms).
 * Frames audio into 256-sample packets (16 ms) and transmits over High-Speed UART (921,600 baud)
 * or USB CDC directly to the Raspberry Pi for GTCRN neural enhancement.
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "driver/i2s_std.h"
#include "driver/uart.h"
#include "driver/gpio.h"

#include "impulse_detector.h"
#include "packet_protocol.h"

static const char *TAG = "ESP32S3_AUDIO_GUARD";

/* GPIO Pin Assignments */
#define I2S_BCLK_PIN         GPIO_NUM_4
#define I2S_WS_PIN           GPIO_NUM_5
#define I2S_DIN_PIN          GPIO_NUM_6

#define UART_PORT_NUM        UART_NUM_1
#define UART_TX_PIN          GPIO_NUM_17
#define UART_RX_PIN          GPIO_NUM_18
#define UART_BAUD_RATE       921600

#define STATUS_LED_PIN       GPIO_NUM_2

/* Audio Specifications */
#define SAMPLE_RATE          16000
#define SAMPLES_PER_FRAME    256
#define DMA_BUFFER_COUNT     4

static i2s_chan_handle_t     rx_channel_handle = NULL;
static QueueHandle_t         audio_packet_queue = NULL;
static impulse_limiter_t     g_limiter;

/**
 * @brief Configure and initialize I2S RX channel with DMA
 */
static esp_err_t init_i2s_rx(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = DMA_BUFFER_COUNT;
    chan_cfg.dma_frame_num = SAMPLES_PER_FRAME;

    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_channel_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_PIN,
            .ws   = I2S_WS_PIN,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_DIN_PIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_channel_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_channel_handle));
    ESP_LOGI(TAG, "I2S RX initialized at %d Hz, 16-bit mono, 256 frame DMA", SAMPLE_RATE);
    return ESP_OK;
}

/**
 * @brief Configure and initialize High-Speed UART for binary transmission
 */
static esp_err_t init_uart(void) {
    uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    ESP_ERROR_CHECK(uart_driver_install(UART_PORT_NUM, 2048, 2048, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_PORT_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_PORT_NUM, UART_TX_PIN, UART_RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    ESP_LOGI(TAG, "High-Speed UART initialized at %d baud on TX=%d, RX=%d", UART_BAUD_RATE, UART_TX_PIN, UART_RX_PIN);
    return ESP_OK;
}

/**
 * @brief Core 0 Task: Reads DMA audio, applies transient limiting, formats packets
 */
static void audio_capture_task(void *pvParameters) {
    int16_t raw_buffer[SAMPLES_PER_FRAME];
    audio_packet_t packet;
    uint8_t seq = 0;
    size_t bytes_read = 0;

    packet.sync1 = PACKET_SYNC_BYTE_1;
    packet.sync2 = PACKET_SYNC_BYTE_2;

    while (1) {
        // Read 256 samples (512 bytes) from I2S DMA. Blocks until available (16 ms)
        esp_err_t ret = i2s_channel_read(rx_channel_handle, raw_buffer, sizeof(raw_buffer), &bytes_read, portMAX_DELAY);
        if (ret == ESP_OK && bytes_read == sizeof(raw_buffer)) {
            bool impulse_triggered = false;

            // In-place instantaneous impulse limiting (<0.1 ms latency)
            impulse_limiter_process(&g_limiter, raw_buffer, SAMPLES_PER_FRAME, &impulse_triggered);

            if (impulse_triggered) {
                gpio_set_level(STATUS_LED_PIN, 1);
            } else if (!g_limiter.in_impulse_state) {
                gpio_set_level(STATUS_LED_PIN, 0);
            }

            // Fill packet
            packet.sequence_id = seq++;
            packet.flags = 0;
            if (impulse_triggered) packet.flags |= FLAG_IMPULSE_TRIGGERED;
            if (g_limiter.in_impulse_state) packet.flags |= FLAG_LIMITER_ACTIVE;

            memcpy(packet.pcm_samples, raw_buffer, sizeof(raw_buffer));

            // Compute CRC over header + payload
            size_t payload_len = sizeof(audio_packet_t) - sizeof(uint16_t);
            packet.crc16 = compute_crc16((const uint8_t *)&packet, payload_len);

            // Send to UART task queue (non-blocking, drop frame if queue full to preserve real-time)
            if (xQueueSend(audio_packet_queue, &packet, 0) != pdPASS) {
                ESP_LOGW(TAG, "Audio packet queue overrun - dropped 1 frame");
            }
        }
    }
}

/**
 * @brief Core 1 Task: Transmits packets over High-Speed UART
 */
static void uart_tx_task(void *pvParameters) {
    audio_packet_t packet;

    while (1) {
        if (xQueueReceive(audio_packet_queue, &packet, portMAX_DELAY) == pdPASS) {
            uart_write_bytes(UART_PORT_NUM, (const char *)&packet, sizeof(audio_packet_t));
        }
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "=== ESP32-S3 Tactical Speech Capture & Impulse Limiter ===");

    // Initialize LED pin for visual indicator
    gpio_reset_pin(STATUS_LED_PIN);
    gpio_set_direction(STATUS_LED_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(STATUS_LED_PIN, 0);

    // Initialize impulse limiter
    impulse_limiter_init(&g_limiter, SAMPLE_RATE);

    // Create queue holding up to 8 packets (~128 ms buffer)
    audio_packet_queue = xQueueCreate(8, sizeof(audio_packet_t));

    // Initialize peripherals
    ESP_ERROR_CHECK(init_uart());
    ESP_ERROR_CHECK(init_i2s_rx());

    // Launch dual-core pinned FreeRTOS tasks
    xTaskCreatePinnedToCore(audio_capture_task, "audio_cap", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(uart_tx_task, "uart_tx", 4096, NULL, 4, NULL, 1);

    ESP_LOGI(TAG, "System operational. Streaming 16 ms frames @ 62.5 Hz to Raspberry Pi.");
}
