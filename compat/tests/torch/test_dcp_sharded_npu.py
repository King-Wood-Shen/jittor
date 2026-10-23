"""Public DCP persists and restores an uneven FSDP2 shard on two NPUs."""

import json
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
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
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


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu", reason="requires real ACL")
def test_dcp_fresh_optimizer_state_restores_rank_local_moments_on_hccl():
    assert jt.world_size == 2 and dist.get_backend() == "hccl"
    assert jt.compiler.build_config.backend == "acl"
    assert jt.flags.backend_fallback == "error"
    jt.core._device_memory_window_start(0)
    rank = dist.get_rank()
    directory = Path(os.environ["RESULT_DIR"]) / "dcp-fresh-optimizer-state"
    source = {
        "optimizer": {
            "state": {"weight": {"exp_avg": torch.tensor([1.0 + rank, 2.0 + rank], device="npu")}},
            "param_groups": [{"params": ["weight"], "lr": 0.01}],
        }
    }
    dcp.save(
        source,
        storage_writer=dcp.FileSystemWriter(directory),
        planner=dcp.default_planner.DefaultSavePlanner(),
    )
    target = {
        "optimizer": {
            "state": {},
            "param_groups": [{"params": ["weight"], "lr": 0.01}],
        }
    }
    dcp.load(
        target,
        storage_reader=dcp.FileSystemReader(directory),
        planner=dcp.default_planner.DefaultLoadPlanner(),
    )
    restored = target["optimizer"]["state"]["weight"]["exp_avg"]
    assert tuple(restored.shape) == (2,)
    np.testing.assert_array_equal(np.asarray(restored.numpy()), [1.0 + rank, 2.0 + rank])
    model = torch.nn.Linear(2, 1, bias=False)
    fully_shard(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer_loss = model(torch.tensor([[1.0, -1.0]], device="npu")).sum()
    optimizer_loss.backward()
    optimizer.step()
    options = StateDictOptions(full_state_dict=False)
    saved_optimizer = get_optimizer_state_dict(model, optimizer, options=options)
    assert saved_optimizer["state"]
    optimizer_directory = Path(os.environ["RESULT_DIR"]) / "dcp-fresh-fsdp2-optimizer"
    dcp.save(
        {"optimizer": saved_optimizer},
        storage_writer=dcp.FileSystemWriter(optimizer_directory),
        planner=dcp.default_planner.DefaultSavePlanner(),
    )
    fresh_optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
    fresh_state = get_optimizer_state_dict(model, fresh_optimizer, options=options)
    assert fresh_state["state"] == {}
    optimizer_target = {"optimizer": fresh_state}
    dcp.load(
        optimizer_target,
        storage_reader=dcp.FileSystemReader(optimizer_directory),
        planner=dcp.default_planner.DefaultLoadPlanner(),
    )
    set_optimizer_state_dict(model, fresh_optimizer, optimizer_target["optimizer"], options=options)
    restored_optimizer = get_optimizer_state_dict(model, fresh_optimizer, options=options)
    for parameter_name, fields in saved_optimizer["state"].items():
        for field_name, value in fields.items():
            restored_value = restored_optimizer["state"][parameter_name][field_name]
            if isinstance(value, jt.Var):
                np.testing.assert_array_equal(
                    np.asarray(restored_value.numpy()), np.asarray(value.numpy())
                )
            else:
                assert restored_value == value
    torch.npu.synchronize()
    fallback = int(jt.core.backend_fallback_count())
    peak_bytes = int(jt.core._device_memory_window_peak(0))
    assert fallback == 0
    assert peak_bytes > 0
    print(
        "DCP_OPTIMIZER_RANK_ACCEPTED "
        + json.dumps(
            {
                "rank": rank,
                "fallback": fallback,
                "peak_bytes": peak_bytes,
                "fsdp2_optimizer_setter": "passed",
            }
        ),
        flush=True,
    )
