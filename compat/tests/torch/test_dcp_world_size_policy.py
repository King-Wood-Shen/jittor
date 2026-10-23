"""DCP only reshares model tensor state to a singleton target."""

from types import SimpleNamespace

import pytest

from jittor.compat.torch import checkpoint_dcp
from jittor.compat.torch.checkpoint_dcp import _validate_load_world_size


def _metadata(world_size, *, rank_local=()):
    return {
        "world_size": world_size,
        "records": [
            {
                "tree": {
                    "kind": "dict",
                    "items": [
                        [key, {"kind": "rank_local" if key in rank_local else "dict"}]
                        for key in ("model", "optimizer")
                    ],
                },
            }
        ],
    }


def test_model_checkpoint_can_reshard_into_singleton():
    _validate_load_world_size(_metadata(2), 1)


def test_rank_local_optimizer_checkpoint_cannot_reshard():
    with pytest.raises(NotImplementedError, match="rank-local state.*optimizer"):
        _validate_load_world_size(_metadata(2, rank_local=("optimizer",)), 1)


def test_other_world_size_changes_remain_unsupported():
    with pytest.raises(NotImplementedError, match="only model-only restore"):
        _validate_load_world_size(_metadata(2), 4)


def test_singleton_load_group_does_not_create_native_store(tmp_path, monkeypatch):
    class SingletonDist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 1

    monkeypatch.setattr(
        checkpoint_dcp, "_torch", lambda: SimpleNamespace(distributed=SingletonDist)
    )
    monkeypatch.setattr(
        checkpoint_dcp.native_dist,
        "get_default_store",
        lambda: pytest.fail("singleton DCP restore must not initialize a native Store"),
    )
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")

    assert checkpoint_dcp._group(tmp_path, allow_singleton=True) is SingletonDist
    assert not (tmp_path / "dcp-native-store.sqlite").exists()
