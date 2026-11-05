"""Real shared CPU epoch storage and multiprocessing ownership boundaries."""

import gc
import multiprocessing as mp
import textwrap
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from jittor.compat.shim import activate

torch = activate()["torch"]

from _helpers.child_process import run_python_child


def _child_write_epoch(epoch, value, result_queue):
    assert epoch.is_shared()
    # datasets.IterableDataset advances its shared worker epoch in place.
    # Exercise the same operator rather than only proving fill_ writes bytes.
    epoch += value
    epoch.sync()
    result_queue.put(int(epoch.item()))


def _child_read_epoch(epoch, result_queue):
    assert epoch.is_shared()
    result_queue.put(int(epoch.item()))


@pytest.mark.parametrize("method", ("fork", "spawn"))
def test_shared_epoch_is_visible_in_both_process_directions(method):
    if method not in mp.get_all_start_methods():
        pytest.skip("multiprocessing start method is unavailable")
    context = mp.get_context(method)
    epoch = torch.tensor(3, dtype=torch.int64, device="cpu")
    alias = epoch
    assert epoch.share_memory_() is alias
    assert epoch.is_shared()
    assert epoch.share_memory_() is epoch

    result_queue = context.Queue()
    process = context.Process(target=_child_write_epoch, args=(epoch, 17, result_queue))
    process.start()
    process.join(120)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("shared epoch writer did not finish")
    assert process.exitcode == 0
    assert result_queue.get(timeout=10) == 20
    assert int(epoch.item()) == 20

    epoch += 12
    epoch.sync()
    process = context.Process(target=_child_read_epoch, args=(epoch, result_queue))
    process.start()
    process.join(120)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("shared epoch reader did not finish")
    assert process.exitcode == 0
    assert result_queue.get(timeout=10) == 32
    result_queue.close()
    result_queue.join_thread()


def test_pending_alias_follows_rebound_storage_and_materialized_alias_is_rejected():
    epoch = torch.tensor(4, dtype=torch.int64, device="cpu")
    pending_alias = epoch.detach()
    assert epoch.share_memory_() is epoch
    pending_alias.sync()
    assert pending_alias.is_shared()
    pending_alias.fill_(8)
    pending_alias.sync()
    assert int(epoch.item()) == 8

    separate = torch.tensor(5, dtype=torch.int64, device="cpu")
    materialized_alias = separate.detach()
    materialized_alias.sync()
    with pytest.raises(RuntimeError, match="materialized view aliases"):
        separate.share_memory_()


def test_new_shared_view_keeps_storage_shared_and_owner_alive():
    epoch = torch.tensor(6, dtype=torch.int64, device="cpu")
    epoch.share_memory_()
    name = epoch._torch_shared_memory_owner.name
    view = epoch.view(1, 1)
    view.sync()
    assert view.is_shared()
    del epoch
    gc.collect()
    attached = SharedMemory(name=name, create=False)
    try:
        assert np.ndarray((1,), dtype=np.int64, buffer=attached.buf)[0] == 6
        view.fill_(11)
        view.sync()
        assert np.ndarray((1,), dtype=np.int64, buffer=attached.buf)[0] == 11
    finally:
        attached.close()


@pytest.mark.parametrize("value,dtype", (([1, 2], torch.int64), (1, torch.int32)))
def test_unsupported_shared_tensor_shapes_and_dtypes_fail_closed(value, dtype):
    with pytest.raises(NotImplementedError, match="one-element int64"):
        torch.tensor(value, dtype=dtype, device="cpu").share_memory_()


def test_shared_segment_is_unlinked_when_its_process_family_exits():
    script = textwrap.dedent("""
        from jittor.compat.shim import activate
        torch = activate()["torch"]
        value = torch.tensor(13, dtype=torch.int64, device="cpu")
        value.share_memory_()
        print("SHM_NAME=" + value._torch_shared_memory_owner.name, flush=True)
    """)
    result = run_python_child(["-c", script], without_torch_mode=True,
                              merge_stderr=True)
    assert result.returncode == 0, result.stdout
    name = next(line.split("=", 1)[1] for line in result.stdout.splitlines()
                if line.startswith("SHM_NAME="))
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name, create=False)


def test_cuda_shared_scalar_is_rejected_explicitly():
    import jittor as jt
    from _helpers.capability import check_accelerator

    if not check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("requires CUDA for the unsupported-device assertion")
    value = torch.tensor(1, dtype=torch.int64, device="cuda")
    with pytest.raises(NotImplementedError, match="CPU tensors only"):
        value.share_memory_()
