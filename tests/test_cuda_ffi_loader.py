from types import SimpleNamespace

import pytest

from mpm_jax.cuda import p2g_cuda


def _isolate_registry(monkeypatch):
    monkeypatch.setattr(p2g_cuda, "_REGISTERED", set())


def test_register_missing_extension_raises(monkeypatch):
    def fake_import_module(name):
        raise ImportError(name)

    monkeypatch.setattr(p2g_cuda.importlib, "import_module", fake_import_module)
    _isolate_registry(monkeypatch)

    with pytest.raises(ImportError):
        p2g_cuda._register("unit_test_missing", "missing_factory")


@pytest.mark.parametrize(
    ("registrar", "target", "factory"),
    [
        (p2g_cuda.register_p2g_inline, p2g_cuda._P2G_INLINE_TARGET, "p2g_inline"),
        (
            p2g_cuda.register_p2g_v2_inline,
            p2g_cuda._P2G_V2_INLINE_TARGET,
            "p2g_v2_inline",
        ),
        (
            p2g_cuda.register_p2g_v3_inline,
            p2g_cuda._P2G_V3_INLINE_TARGET,
            "p2g_v3_inline",
        ),
        (
            p2g_cuda.register_p2g_v4_inline,
            p2g_cuda._P2G_V4_INLINE_TARGET,
            "p2g_v4_inline",
        ),
    ],
)
def test_public_registrar_imports_expected_capsule(monkeypatch, registrar, target, factory):
    capsules = {
        "p2g_inline": object(),
        "p2g_v2_inline": object(),
        "p2g_v3_inline": object(),
        "p2g_v4_inline": object(),
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

    assert registrar()

    assert calls == [
        (
            target,
            capsules[factory],
            {"platform": "CUDA", "api_version": 1},
        )
    ]


def test_register_is_cached(monkeypatch):
    capsule = object()
    imported = []

    fake_module = SimpleNamespace(p2g_inline=lambda: capsule)
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

    name = "unit_test_cache_check"
    assert p2g_cuda._register(name, "p2g_inline")
    assert p2g_cuda._register(name, "p2g_inline")
    assert imported == [p2g_cuda._FFI_MODULE]
