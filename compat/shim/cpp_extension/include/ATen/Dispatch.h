// ATen floating dispatch for extensions built by the Jittor ABI shim.
#pragma once
#include <c10/util/Half.h>

// Dispatch is an explicit CUDA-toolkit header surface. Keep it out of
// torch/extension.h, whose CUDA-free contract lets extensions define float3.
// The SDK type supplies the existing host/device FP16 representation and math.

namespace jtorch { namespace detail {
template <ScalarType> struct DispatchScalar;
template <> struct DispatchScalar<ScalarType::Float> { using type = float; };
template <> struct DispatchScalar<ScalarType::Double> { using type = double; };
template <> struct DispatchScalar<ScalarType::Half> { using type = c10::Half; };
}}

#define AT_DISPATCH_CASE(enum_type, ...) \
    case enum_type: { \
        using scalar_t = typename ::jtorch::detail::DispatchScalar<enum_type>::type; \
        return __VA_ARGS__(); \
    }

#define AT_DISPATCH_SWITCH(TYPE, NAME, ...) \
    [&] { \
        const auto& the_type = (TYPE); \
        constexpr const char* at_dispatch_name = NAME; \
        const auto _st = static_cast<::at::ScalarType>(the_type); \
        switch (_st) { \
            __VA_ARGS__ \
            default: \
                throw std::runtime_error(::jtorch::detail::_concat( \
                    at_dispatch_name, " not implemented for scalar type ", static_cast<int>(_st))); \
        } \
    }()

#define AT_DISPATCH_CASE_FLOATING_TYPES(...) \
    AT_DISPATCH_CASE(::at::ScalarType::Double, __VA_ARGS__) \
    AT_DISPATCH_CASE(::at::ScalarType::Float, __VA_ARGS__)
#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(...) \
    AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__) \
    AT_DISPATCH_CASE(::at::ScalarType::Half, __VA_ARGS__)
#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(__VA_ARGS__))
