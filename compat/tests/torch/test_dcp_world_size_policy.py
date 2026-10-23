"""DCP only reshares model tensor state to a singleton target."""

import pytest

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
