"""Public DCP persists and restores an uneven FSDP2 shard on two NPUs."""

import os
from pathlib import Path

import jittor as jt
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from jittor.serialization.sharded import read_manifest
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.fsdp import fully_shard


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu", reason="requires real ACL")
def test_dcp_uneven_fsdp_shards_roundtrip_on_hccl():
    assert jt.world_size == 2 and dist.get_backend() == "hccl"
    assert jt.compiler.build_config.backend == "acl"
    assert jt.flags.backend_fallback == "error"
    jt.core._device_memory_window_start(0)
    jt.set_global_seed(13)

    model = torch.nn.Linear(5, 3, bias=False)
    fully_shard(model)
    options = StateDictOptions(full_state_dict=False)
    local = get_model_state_dict(model, options=options)["weight"]
    before = np.array(local.numpy(), copy=True)
    directory = Path(os.environ["JITTOR_DIST_RENDEZVOUS_DIR"]) / "dcp-shard-regression"

    dcp.save(
        {"model": {"weight": local}},
        storage_writer=dcp.FileSystemWriter(directory),
        planner=dcp.default_planner.DefaultSavePlanner(),
    )
    manifest = read_manifest(directory)
    assert manifest["world_size"] == 2
    intervals = sorted(
        (record["tensors"]["/model/weight"]["offset"], record["tensors"]["/model/weight"]["count"])
        for record in manifest["records"]
    )
    assert intervals == [(0, 8), (8, 7)]

    local.assign(torch.zeros_like(local))
    assert not np.array_equal(np.asarray(local.numpy()), before)
    target = {"model": get_model_state_dict(model, options=options)}
    dcp.load(
        target,
        storage_reader=dcp.FileSystemReader(directory),
        planner=dcp.default_planner.DefaultLoadPlanner(),
    )
    set_model_state_dict(model, target["model"], options=options)
    restored = np.asarray(get_model_state_dict(model, options=options)["weight"].numpy())
    np.testing.assert_array_equal(restored, before)
    torch.npu.synchronize()
    assert jt.core.backend_fallback_count() == 0
    assert jt.core._device_memory_window_peak(0) > 0
