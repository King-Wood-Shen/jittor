"""Owners for the bounded shared CPU epoch-tensor compatibility path."""

from multiprocessing.shared_memory import SharedMemory


class SharedMemoryOwner:
    """Keep one local mapping open; multiprocessing's tracker owns unlink."""

    __slots__ = ("handle",)

    def __init__(self, handle):
        self.handle = handle

    @property
    def name(self):
        return self.handle.name

    def unlink_after_failed_install(self):
        self.handle.unlink()


def create_epoch_buffer(value):
    import numpy as np

    handle = SharedMemory(create=True, size=np.dtype(np.int64).itemsize)
    owner = SharedMemoryOwner(handle)
    array = np.ndarray((1,), dtype=np.int64, buffer=handle.buf)
    array[0] = value
    return owner, array


def attach_epoch_buffer(name):
    import numpy as np

    handle = SharedMemory(name=name, create=False)
    owner = SharedMemoryOwner(handle)
    array = np.ndarray((1,), dtype=np.int64, buffer=handle.buf)
    return owner, array
