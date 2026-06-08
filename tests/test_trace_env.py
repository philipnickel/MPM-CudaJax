import os

from simulate import _disable_command_buffers_for_trace


def test_trace_config_disables_command_buffers_before_jax_backend_init(monkeypatch):
    monkeypatch.setenv(
        "XLA_FLAGS",
        "--foo=bar --xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,WHILE",
    )

    changed = _disable_command_buffers_for_trace(["simulate.py", "-cn", "trace"])

    assert changed is True
    assert os.environ["XLA_FLAGS"] == "--foo=bar --xla_gpu_enable_command_buffer="


def test_profile_enabled_disables_command_buffers(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)

    changed = _disable_command_buffers_for_trace(
        ["simulate.py", "profile.enabled=true"]
    )

    assert changed is True
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_enable_command_buffer="


def test_trace_command_buffer_disable_can_be_overridden(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer=FUSION")

    changed = _disable_command_buffers_for_trace(
        ["simulate.py", "-cn", "trace", "profile.disable_command_buffers=false"]
    )

    assert changed is False
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_enable_command_buffer=FUSION"
