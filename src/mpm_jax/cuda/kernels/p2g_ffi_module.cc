#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;

XLA_FFI_DECLARE_HANDLER_SYMBOL(P2GInline);
XLA_FFI_DECLARE_HANDLER_SYMBOL(P2GV2Inline);
XLA_FFI_DECLARE_HANDLER_SYMBOL(P2GV3Inline);
XLA_FFI_DECLARE_HANDLER_SYMBOL(P2GV4Inline);

template <typename T>
nb::capsule EncapsulateFfiHandler(T* fn) {
    static_assert(
        std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
        "Encapsulated function must be an XLA FFI handler"
    );
    return nb::capsule(reinterpret_cast<void*>(fn));
}

NB_MODULE(_p2g_ffi, m) {
    m.def("p2g_inline", []() { return EncapsulateFfiHandler(P2GInline); });
    m.def("p2g_v2_inline", []() { return EncapsulateFfiHandler(P2GV2Inline); });
    m.def("p2g_v3_inline", []() { return EncapsulateFfiHandler(P2GV3Inline); });
    m.def("p2g_v4_inline", []() { return EncapsulateFfiHandler(P2GV4Inline); });
}
