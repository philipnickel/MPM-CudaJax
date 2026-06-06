#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;

XLA_FFI_DECLARE_HANDLER_SYMBOL(P2GV1);
XLA_FFI_DECLARE_HANDLER_SYMBOL(P2GV2);
XLA_FFI_DECLARE_HANDLER_SYMBOL(P2GV3);
XLA_FFI_DECLARE_HANDLER_SYMBOL(P2GV4);

template <typename T>
nb::capsule EncapsulateFfiHandler(T* fn) {
    static_assert(
        std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
        "Encapsulated function must be an XLA FFI handler"
    );
    return nb::capsule(reinterpret_cast<void*>(fn));
}

NB_MODULE(_p2g_ffi, m) {
    m.def("p2g_v1", []() { return EncapsulateFfiHandler(P2GV1); });
    m.def("p2g_v2", []() { return EncapsulateFfiHandler(P2GV2); });
    m.def("p2g_v3", []() { return EncapsulateFfiHandler(P2GV3); });
    m.def("p2g_v4", []() { return EncapsulateFfiHandler(P2GV4); });
}
