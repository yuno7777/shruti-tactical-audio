idf_component_register(
    SRCS "main.c" "impulse_detector.c"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_i2s esp_driver_uart esp_driver_gpio
)
