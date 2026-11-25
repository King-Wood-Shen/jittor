#pragma once
#include <cstdint>
#include <cstring>

namespace jittor {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define JT_FP8_HD __host__ __device__
#else
#define JT_FP8_HD
#endif

// E4M3FN uses all exponent codes for finite values except magnitude 0x7f.
// Store the exact byte; conversions round to nearest, ties to even.
struct float8_e4m3fn {
    std::uint8_t bits;

    JT_FP8_HD float8_e4m3fn() = default;

    JT_FP8_HD explicit float8_e4m3fn(float value) {
        const auto raw = float_bits(value);
        const auto sign = (raw >> 24) & 0x80u;
        const int exponent = int((raw >> 23) & 0xffu) - 127;
        const auto fraction = raw & 0x7fffffu;
        unsigned magnitude = 0;
        if (exponent > 8) {
            magnitude = 0x7f;
        } else if (exponent >= -6) {
            magnitude = unsigned(exponent + 7) * 8 + rounded_shift(fraction, 20);
        } else if (exponent >= -10) {
            magnitude = rounded_shift(fraction | 0x800000u, 14 - exponent);
        }
        bits = std::uint8_t(sign | (magnitude < 0x7f ? magnitude : 0x7f));
    }

    JT_FP8_HD operator float() const {
        const std::uint32_t sign = std::uint32_t(bits & 0x80u) << 24;
        const unsigned magnitude = bits & 0x7fu;
        if (magnitude == 0x7f)
            return from_float_bits(sign | 0x7fc00000u);
        if (magnitude < 8) {
            auto raw = float_bits(float(magnitude) * (1.0f / 512.0f));
            return from_float_bits(sign | raw);
        }
        return from_float_bits(sign | ((magnitude / 8 + 120u) << 23)
                                    | ((magnitude & 7u) << 20));
    }

private:
    JT_FP8_HD static unsigned rounded_shift(unsigned value, int shift) {
        const unsigned quotient = value >> shift;
        return (value + (1u << (shift - 1)) - 1u + (quotient & 1u)) >> shift;
    }

    JT_FP8_HD static std::uint32_t float_bits(float value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return __float_as_uint(value);
#else
        std::uint32_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        return raw;
#endif
    }

    JT_FP8_HD static float from_float_bits(std::uint32_t raw) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return __uint_as_float(raw);
#else
        float value;
        std::memcpy(&value, &raw, sizeof(value));
        return value;
#endif
    }
};

#undef JT_FP8_HD
static_assert(sizeof(float8_e4m3fn) == 1, "E4M3FN must occupy one byte");

} // namespace jittor
