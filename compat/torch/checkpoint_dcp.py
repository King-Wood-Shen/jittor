"""Torch DCP's public entry points over the native tensor-shard checkpoint.

The native owner writes, validates and atomically publishes rank files. This
layer only translates FSDP2's local state into its TensorShard descriptors and
loads an unchanged one-node world into the caller's preallocated state dict.
"""

import os
from pathlib import Path

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._core.var import placement_scope_like
from jittor import distributed as native_dist
from jittor.distributed import checkpoint as native_checkpoint
from jittor.distributed.store import FileStore
from jittor.serialization.sharded import RankLocal, TensorShard, read_manifest

from .. import fsdp_hooks
from .context import get_install_context


class DefaultSavePlanner:
    """Use the native owner's complete rank-file transaction."""

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise NotImplementedError("custom DCP save planning is not supported")


class DefaultLoadPlanner:
    """Restore only the current rank's validated local intervals."""

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise NotImplementedError("custom DCP load planning is not supported")


def _torch():
    return get_install_context(jt).target_namespace


def _group(directory):
    dist = _torch().distributed
    if not dist.is_initialized():
        raise RuntimeError("DCP sharded save/load requires an initialized process group")
    world = int(dist.get_world_size())
    if world < 2:
        raise NotImplementedError("DCP rank-local storage requires at least two ranks")
    if (
        int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world
        or int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) != 1
    ):
        raise NotImplementedError("DCP supports one node and an unchanged world only")
    if native_dist.get_default_store() is None:
        rendezvous = Path(os.environ.get("JITTOR_DIST_RENDEZVOUS_DIR", directory))
        if rendezvous.is_symlink():
            raise ValueError("DCP rendezvous directory cannot be a symlink")
        rendezvous.mkdir(parents=True, exist_ok=True)
        store = FileStore(rendezvous / "dcp-native-store.sqlite", world)
        try:
            native_dist.init_process_group(
                backend=dist.get_backend(),
                store=store,
                rank=int(dist.get_rank()),
                world_size=world,
            )
        except Exception:
            store.close()
            raise
    return dist


def _storage(storage, kind):
    if storage is None or type(storage).__name__ != "FileSystem" + kind:
        raise NotImplementedError("DCP requires FileSystem" + kind)
    return Path(storage.path)


def _shard(value):
    if not isinstance(value, jt.Var) or not hasattr(value, "_jittor_fsdp2_entry"):
        return None
    provider = fsdp_hooks.provider()
    if provider is None:
        raise RuntimeError("FSDP2 shard has no registered checkpoint provider")
    return provider._dcp_shard_descriptor(value)


def _save_tree(value):
    if isinstance(value, jt.Var):
        return _shard(value) or value
    if isinstance(value, dict):
        return {key: _save_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_save_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_save_tree(item) for item in value)
    return value


def _pointer(path, key):
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _load_targets(value, path=""):
    if isinstance(value, jt.Var):
        shard = _shard(value)
        return {path: shard} if shard is not None else {}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(_load_targets(item, _pointer(path, key)))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(_load_targets(item, _pointer(path, index)))
        return result
    return {}


def _saved_tensor(value):
    return value.tensor if isinstance(value, TensorShard) else value


def _check_tree(current, saved, path=""):
    if isinstance(current, jt.Var):
        source = _saved_tensor(saved)
        if not isinstance(source, jt.Var) or dtype_name(source.dtype) != dtype_name(current.dtype):
            raise RuntimeError("DCP tensor type or dtype changed: " + path)
        if isinstance(saved, TensorShard):
            if (
                len(current.shape) != 1
                or len(source.shape) != 1
                or source.shape[0] > current.shape[0]
            ):
                raise RuntimeError("DCP local shard shape changed: " + path)
        elif tuple(source.shape) != tuple(current.shape):
            raise RuntimeError("DCP tensor shape changed: " + path)
        return
    if isinstance(current, dict):
        if not isinstance(saved, dict) or set(current) != set(saved):
            raise RuntimeError("DCP state keys changed: " + path)
        for key, value in current.items():
            _check_tree(value, saved[key], _pointer(path, key))
        return
    if isinstance(current, (list, tuple)):
        if type(current) is not type(saved) or len(current) != len(saved):
            raise RuntimeError("DCP state sequence changed: " + path)
        for index, value in enumerate(current):
            _check_tree(value, saved[index], _pointer(path, index))


def _assign_tree(current, saved):
    if isinstance(current, jt.Var):
        source = _saved_tensor(saved)
        with placement_scope_like(current):
            source = _torch().tensor(source, dtype=current.dtype, device=current.device).detach()
            if isinstance(saved, TensorShard) and source.shape[0] < current.shape[0]:
                padding = _torch().zeros(
                    (int(current.shape[0]) - int(source.shape[0]),),
                    dtype=current.dtype,
                    device=current.device,
                )
                source = _torch().cat((source, padding), dim=0)
        current.assign(source)
        return current
    if isinstance(current, dict):
        for key, value in current.items():
            current[key] = _assign_tree(value, saved[key])
        return current
    if isinstance(current, list):
        for index, value in enumerate(current):
            current[index] = _assign_tree(value, saved[index])
        return current
    if isinstance(current, tuple):
        return tuple(_assign_tree(value, saved[index]) for index, value in enumerate(current))
    return saved


def save(state_dict, *, storage_writer=None, planner=None, **kwargs):
    if kwargs:
        raise NotImplementedError("DCP save options are unsupported: " + ", ".join(sorted(kwargs)))
    if planner is not None and not isinstance(planner, DefaultSavePlanner):
        raise NotImplementedError("custom DCP save planner is unsupported")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("DCP save expects a nonempty state dictionary")
    directory = _storage(storage_writer, "Writer")
    _group(directory)
    converted = {
        key: RankLocal(value) if key == "optimizer" else _save_tree(value)
        for key, value in state_dict.items()
    }
    return native_checkpoint.save(converted, directory)


def load(state_dict, *, storage_reader=None, planner=None, **kwargs):
    if kwargs:
        raise NotImplementedError("DCP load options are unsupported: " + ", ".join(sorted(kwargs)))
    if planner is not None and not isinstance(planner, DefaultLoadPlanner):
        raise NotImplementedError("custom DCP load planner is unsupported")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("DCP load expects a preallocated nonempty state dictionary")
    directory = _storage(storage_reader, "Reader")
    dist = _group(directory)
    metadata = read_manifest(directory)
    if metadata["world_size"] != int(dist.get_world_size()):
        raise NotImplementedError("DCP checkpoint world size differs; resharding is unsupported")
    restored = native_checkpoint.load(directory, tensor_shards=_load_targets(state_dict))
    _check_tree(state_dict, restored)
    _assign_tree(state_dict, restored)
    return None
