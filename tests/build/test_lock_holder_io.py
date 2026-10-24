"""Lock-holder diagnostics must work without POSIX positional-I/O APIs."""

import os
import tempfile
import unittest
from unittest import mock

from jittor_utils import lock


class TestLockHolderPortableIO(unittest.TestCase):
    def _assert_holder_record_preserves_fd_position(self, path):
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.write(fd, b"preserve this descriptor position")
            position = 7
            os.lseek(fd, position, os.SEEK_SET)

            lock._write_holder(fd)
            description = lock.describe_holder(fd)

            self.assertIn("held by pid {}".format(os.getpid()), description)
            self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), position)
        finally:
            os.close(fd)

    def test_holder_record_uses_platform_positional_io(self):
        with tempfile.TemporaryDirectory() as directory:
            self._assert_holder_record_preserves_fd_position(
                os.path.join(directory, "jittor.lock"))

    def test_holder_record_falls_back_without_positional_io(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "jittor.lock")
            # Windows Python does not expose either positional operation.
            # Setting them to None exercises that same feature probe on POSIX.
            with mock.patch.object(lock.os, "pwrite", None, create=True), \
                    mock.patch.object(lock.os, "pread", None, create=True):
                self._assert_holder_record_preserves_fd_position(path)


if __name__ == "__main__":
    unittest.main()
