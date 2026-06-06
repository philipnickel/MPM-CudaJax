from types import SimpleNamespace

from mpm_jax.cuda import p2g_cuda


def test_register_missing_extension_returns_false(monkeypatch):
    """When the native extension is absent we should fail gracefully."""

    def fake_import_module(name):
        raise ImportError(name)

    monkeypatch.setattr(p2g_cuda.importlib, "import_module", fake_import_module)
    p2g_cuda._REGISTERED.clear()

    assert p2g_cuda._register("unit_test_missing", "missing_factory") is False


def test_register_imports_extension_capsule_and_calls_ffi(monkeypatch):
    capsule = object()
    calls = {}

    fake_module = SimpleNamespace(p2g_inline=lambda: capsule)
    monkeypatch.setattr(
        p2g_cuda.importlib,
        "import_module",
        lambda name: fake_module,
    )

    def fake_register_ffi_target(name, fn, **kwargs):
        calls["name"] = name
        calls["fn"] = fn
        calls.update(kwargs)

    monkeypatch.setattr(
        p2g_cuda.jax.ffi, "register_ffi_target", fake_register_ffi_target
    )
    p2g_cuda._REGISTERED.clear()

    assert p2g_cuda._register("unit_test_p2g_inline_cuda", "p2g_inline")

    assert calls["name"] == "unit_test_p2g_inline_cuda"
    assert calls["fn"] is capsule
    assert calls["platform"] == "CUDA"
    assert calls["api_version"] == 1


def test_register_is_cached(monkeypatch):
    capsule = object()
    imported = []

    fake_module = SimpleNamespace(p2g_inline=lambda: capsule)
    monkeypatch.setattr(
        p2g_cuda.importlib,
        "import_module",
        lambda name: (imported.append(name), fake_module)[1],
    )
    monkeypatch.setattr(p2g_cuda.jax.ffi, "register_ffi_target", lambda *a, **k: None)
    p2g_cuda._REGISTERED.clear()

    name = "unit_test_cache_check"
    assert p2g_cuda._register(name, "p2g_inline")
    assert p2g_cuda._register(name, "p2g_inline")
    assert imported == [p2g_cuda._FFI_MODULE]
