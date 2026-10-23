"""Native training checkpoints retain the active ACL random stream."""

import os
import sys

import numpy as np
import pytest

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.distributed import checkpoint
from _helpers.native_checkpoint import model_optimizer, training_step


@pytest.mark.skipif(
    os.environ.get("JITTOR_TEST_DEVICES") != "npu", reason="requires a selected ACL device"
)
def test_acl_checkpoint_rng_and_adam_continuation(tmp_path):
    assert "torch" not in sys.modules
    assert jt.compiler.build_config.backend == "acl"
    with jt.flag_scope(use_cuda=1, use_acl=1), forbid_backend_fallbacks():
        jt.core.reset_device_memory_peak(0)
        jt.set_seed(20261023)
        model, optimizer = model_optimizer()
        before = model.weight.numpy().copy()
        losses = [training_step(model, optimizer)["loss"] for _ in range(3)]
        assert np.isfinite(losses).all() and losses[-1] < losses[0]
        assert not np.array_equal(model.weight.numpy(), before)

        checkpoint.save_training_state(model, optimizer, tmp_path)
        saved = checkpoint.load(tmp_path)
        assert saved["rng"]["cuda"] == []
        assert len(saved["rng"]["acl"]) == jt.core.backend_device_count("acl")

        expected_random = jt.rand((13,)).numpy().copy()
        expected_step = training_step(model, optimizer)
        fresh, resumed = model_optimizer()
        assert checkpoint.load_training_state(fresh, resumed, tmp_path) is None
        np.testing.assert_array_equal(jt.rand((13,)).numpy(), expected_random)
        actual_step = training_step(fresh, resumed)
        for name, value in expected_step["model"].items():
            np.testing.assert_allclose(actual_step["model"][name], value, rtol=2e-5, atol=2e-6)
        assert actual_step["steps"] == expected_step["steps"]

        current_acl = jt.core.rng_states("acl")
        current_cpu = jt.core.rng_state("cpu")
        with pytest.raises(ValueError, match="backend mismatch"):
            checkpoint._restore_rng(dict(saved["rng"], acl=[]))
        with pytest.raises(ValueError, match="device count"):
            checkpoint._restore_rng(dict(saved["rng"], acl=saved["rng"]["acl"] * 2))
        assert jt.core.rng_states("acl") == current_acl
        assert jt.core.rng_state("cpu") == current_cpu
        assert jt.core.backend_fallback_count() == 0
        assert jt.core.device_memory_peak(0) > 0
