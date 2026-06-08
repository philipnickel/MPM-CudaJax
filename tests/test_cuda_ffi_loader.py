from types import SimpleNamespace

import pytest

from mpm_jax.p2g.cuda import p2g_cuda


def _isolate_registry(monkeypatch):
    monkeypatch.setattr(p2g_cuda, "_REGISTERED", set())


def test_register_missing_extension_raises(monkeypatch):
    def fake_import_module(name):
        raise ImportError(name)

    monkeypatch.setattr(p2g_cuda.importlib, "import_module", fake_import_module)
    _isolate_registry(monkeypatch)

    with pytest.raises(ImportError):
        p2g_cuda.CudaV1P2G()


@pytest.mark.parametrize(
    ("kernel_type", "factory"),
    [
        (p2g_cuda.CudaV1P2G, "p2g_v1"),
        (p2g_cuda.CudaV3P2G, "p2g_v3"),
        (p2g_cuda.CudaV4P2G, "p2g_v4"),
    ],
)
def test_kernel_class_imports_expected_capsule(monkeypatch, kernel_type, factory):
    capsules = {
        "p2g_v1": object(),
        "p2g_v3": object(),
        "p2g_v4": object(),
    }
    calls = []

    fake_module = SimpleNamespace(
        **{name: (lambda name=name: capsules[name]) for name in capsules}
    )
    monkeypatch.setattr(
        p2g_cuda.importlib,
        "import_module",
        lambda name: fake_module,
    )

    def fake_register_ffi_target(name, fn, **kwargs):
        calls.append((name, fn, kwargs))

    monkeypatch.setattr(
        p2g_cuda.jax.ffi, "register_ffi_target", fake_register_ffi_target
    )
    _isolate_registry(monkeypatch)

    kernel = kernel_type()

    assert kernel.target in p2g_cuda._REGISTERED
    assert calls == [
        (
            kernel_type.target,
            capsules[factory],
            {"platform": "CUDA", "api_version": 1},
        )
    ]


def test_register_is_cached(monkeypatch):
    capsule = object()
    imported = []

    fake_module = SimpleNamespace(p2g_v1=lambda: capsule)
    monkeypatch.setattr(
        p2g_cuda.importlib,
        "import_module",
        lambda name: (imported.append(name), fake_module)[1],
    )
    monkeypatch.setattr(
        p2g_cuda.jax.ffi,
        "register_ffi_target",
        lambda *_args, **_kwargs: None,
    )
    _isolate_registry(monkeypatch)

    p2g_cuda.CudaV1P2G()
    p2g_cuda.CudaV1P2G()
    assert imported == [p2g_cuda._FFI_MODULE]
