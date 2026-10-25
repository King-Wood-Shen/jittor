# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import os
import json
import subprocess
import tempfile
import time
import traceback
import jittor as jt
import jittor_utils as jit_utils
from jittor_utils import lock as jit_lock

from _helpers.child_process import PYTHON, child_env, run_python_child, shell_status


class TestDisableLockOptIn(unittest.TestCase):
    def test_disabling_the_lock_warns_in_a_fresh_process(self):
        result = run_python_child(
            ["-c", "import jittor_utils.lock"],
            env={"JT_BUILD_DISABLE_LOCK": "1"},
            text=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(
            "JT_BUILD_DISABLE_LOCK=1: Jittor build locking is disabled",
            result.stdout.decode(),
        )

    def test_the_deprecated_unprefixed_name_still_disables_the_lock(self):
        # 2.22 gave every setting a namespace; the old name has to keep working
        # or every existing script and gate that uses it breaks at once.
        result = run_python_child(
            ["-c", "import jittor_utils.lock"],
            env={"disable_lock": "1"},
            text=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(
            "JT_BUILD_DISABLE_LOCK=1: Jittor build locking is disabled",
            result.stdout.decode(),
        )

class TestLock(unittest.TestCase):
    def test(self):
        if os.environ.get('lock_full_test', '0') == '1':
            cache_path = os.path.join(jit_utils.home(), ".cache", "jittor", "lock")
            assert os.system(f"rm -rf {cache_path}") == 0
            cmd = f"cache_name=lock {PYTHON} -m jittor.selftest"
        else:
            cmd = f"{PYTHON} -m jittor.selftest"
        print("run cmd twice", cmd)
        assert shell_status(f"{cmd} & {cmd} & wait %1 && wait %2") == 0


# A bare Python child: it must not import jittor, or it would block on the very
# lock these tests hold.
_PROBE = """
import fcntl, sys
f = open(sys.argv[1], 'r+')
try:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('free')
except OSError:
    print('busy')
"""

# The holder exits on a line from stdin rather than being killed: Jittor
# installs a SIGCHLD handler that treats a killed child as an out-of-memory
# event and quick-exits the whole process, so a test that kills a subprocess
# takes the test runner down with it.
_HOLDER = """
import fcntl, json, os, sys, time
f = open(sys.argv[1], 'r+')
fcntl.flock(f, fcntl.LOCK_EX)
f.seek(0); f.truncate()
f.write(json.dumps({"pid": os.getpid(), "time": time.time(), "cmd": "holder"}))
f.flush()
sys.stdout.write("locked\\n"); sys.stdout.flush()
sys.stdin.readline()
"""


def _lock_state(path):
    out = run_python_child(["-c", _PROBE, path], text=False)
    return out.stdout.decode().strip()


@unittest.skipIf(os.name == 'nt', "flock semantics are POSIX only")
@unittest.skipIf(os.environ.get("disable_lock", "0") == "1",
                 "the build lock is disabled in this environment")
class TestBuildLockIsShared(unittest.TestCase):
    """The Python and the C++ side must hold *one* lock, of one kind, on one fd.

    They used to hold two: ``fcntl.flock`` (BSD) from Python and
    ``fcntl(F_SETLKW)`` (POSIX record lock) from C++, on two descriptors. On
    Linux those two families are independent, so both sides could be inside the
    "exclusive" section at once and compile into the same cache directory.
    Every assertion below is about a *third* process observing what we hold:
    that is the only way to tell which lock family is actually in use.

    Nothing here asserts that the file is *idle*, because it is the session's
    lock and the gate runs several workers against one ``JITTOR_HOME``; an
    "idle" assertion would be a statement about the whole session that this
    process cannot make. The two that existed were removed for that reason --
    see the comment in the first case.
    """

    def setUp(self):
        self.lock = jit_lock.jittor_lock
        if self.lock.core is None:
            self.skipTest("jittor_core was not handed the lock descriptor")

    def test_core_lock_is_visible_to_other_processes(self):
        # Deliberately *not* `assertEqual(_lock_state(...), "free")` first, and
        # not at the end either. The file is the session's build lock, shared
        # with every other worker, and the gate runs four of them against one
        # `JITTOR_HOME`: "nobody holds it" is a property of the whole session,
        # not something one process can establish. Measured: with any concurrent
        # holder the probe answers "busy", which is exactly how this failed in
        # the native half of the pull-request gate (`'busy' != 'free'`, while the
        # same file passed serially). What the case is about is the class
        # docstring's claim -- a third process observes *our* hold, and our own
        # release is asserted through `is_locked` below.
        jt.core.lock_acquire()
        try:
            self.assertTrue(jt.core.lock_is_held())
            # One flag, not one per language.
            self.assertTrue(self.lock.is_locked)
            self.assertEqual(_lock_state(self.lock.filename), "busy")
        finally:
            jt.core.lock_release()
        self.assertFalse(self.lock.is_locked)

    def test_closing_another_descriptor_does_not_release_the_lock(self):
        """A POSIX record lock dies when *any* fd for the file is closed."""
        jt.core.lock_acquire()
        try:
            extra = os.open(self.lock.filename, os.O_RDWR)
            os.close(extra)
            self.assertEqual(_lock_state(self.lock.filename), "busy")
        finally:
            jt.core.lock_release()

    def test_python_lock_scope_is_the_same_lock(self):
        with jit_lock.lock_scope():
            self.assertTrue(jt.core.lock_is_held())
            self.assertEqual(_lock_state(self.lock.filename), "busy")
        self.assertFalse(jt.core.lock_is_held())

    def test_nested_scopes_do_not_release_early(self):
        with jit_lock.lock_scope():
            with jit_lock.lock_scope():
                pass
            self.assertEqual(_lock_state(self.lock.filename), "busy")

    @unittest.skipUnless(hasattr(os, "fork"), "fork is required")
    def test_forked_children_get_independent_lock_descriptions(self):
        """Inherited flock descriptors must be reopened before worker locks."""
        if self.lock.is_locked:
            self.skipTest("the parent already holds the session build lock")

        original_fd = self.lock.fd
        original_path = self.lock.filename
        children = []
        child_statuses = []
        with tempfile.TemporaryDirectory() as directory:
            lock_path = os.path.join(directory, "jittor.lock")
            start_path = os.path.join(directory, "start")
            release_path = os.path.join(directory, "release")
            new_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            self.lock.fd = new_fd
            self.lock.filename = lock_path
            self.lock.core.set_lock_fd(new_fd, False)

            def wait_for(path, timeout=10):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if os.path.exists(path):
                        return True
                    time.sleep(0.01)
                return False

            try:
                for worker_id in range(2):
                    pid = os.fork()
                    if pid == 0:
                        try:
                            import jittor_utils
                            jittor_utils.cc.init_subprocess()
                            jt.jt_init_subprocess()
                            if self.lock.fd == new_fd:
                                raise AssertionError("child kept the inherited fd")
                            open(os.path.join(directory, f"ready-{worker_id}"), "w").close()
                            if not wait_for(start_path):
                                raise TimeoutError("parent did not start the workers")
                            with jit_lock.lock_scope():
                                with jit_lock.lock_scope():
                                    pass
                                with open(os.path.join(directory, f"state-{worker_id}"), "w") as state:
                                    state.write("held=%s disable=%s fd=%s inode=%s\n" % (
                                        jt.core.lock_is_held(), jt.flags.disable_lock,
                                        self.lock.fd, os.fstat(self.lock.fd).st_ino))
                                open(os.path.join(directory, f"acquired-{worker_id}"), "w").close()
                                if not wait_for(release_path):
                                    raise TimeoutError("parent did not release workers")
                        except BaseException:
                            with open(os.path.join(directory, f"error-{worker_id}"), "w") as error:
                                error.write(traceback.format_exc())
                            os._exit(1)
                        os._exit(0)
                    children.append(pid)

                self.assertTrue(all(wait_for(os.path.join(directory, f"ready-{i}"))
                                    for i in range(2)), "children did not initialize")
                open(start_path, "w").close()
                acquired_paths = [os.path.join(directory, f"acquired-{i}")
                                  for i in range(2)]
                error_paths = [os.path.join(directory, f"error-{i}")
                               for i in range(2)]

                def acquired_workers():
                    return [i for i, path in enumerate(acquired_paths)
                            if os.path.exists(path)]

                def child_errors():
                    return [(i, open(path).read()) for i, path in enumerate(error_paths)
                            if os.path.exists(path)]

                # Observe either worker against one deadline. Serially waiting
                # for acquired-0 before checking acquired-1 can consume the
                # child's whole release timeout when worker 1 wins first.
                acquisition_deadline = time.monotonic() + 10
                while True:
                    errors = child_errors()
                    if errors:
                        self.fail("forked worker failed before acquisition: " + repr(errors))
                    acquired = acquired_workers()
                    if acquired:
                        break
                    remaining = acquisition_deadline - time.monotonic()
                    if remaining <= 0:
                        self.fail("no child acquired the lock before the deadline")
                    time.sleep(min(0.01, remaining))

                # Keep whichever child acquired first inside lock_scope while
                # its sibling attempts acquisition. The parent publishes
                # release_path only in finally, after checking both markers.
                overlap_deadline = time.monotonic() + 1.0
                while time.monotonic() < overlap_deadline:
                    errors = child_errors()
                    if errors:
                        self.fail("forked worker failed while lock was held: " + repr(errors))
                    if len(acquired_workers()) == 2:
                        break
                    time.sleep(0.01)
                states = {
                    i: (open(os.path.join(directory, f"state-{i}")).read().strip()
                        if os.path.exists(os.path.join(directory, f"state-{i}"))
                        else "not acquired")
                    for i in range(2)
                }
                self.assertEqual(
                    sum(os.path.exists(os.path.join(directory, f"acquired-{i}"))
                        for i in range(2)),
                    1,
                    "forked workers entered the build critical section together: " +
                    repr(states),
                )
            finally:
                open(start_path, "a").close()
                open(release_path, "a").close()
                if self.lock.is_locked:
                    self.lock.unlock()
                for pid in children:
                    _, status = os.waitpid(pid, 0)
                    child_statuses.append((pid, status))
                self.lock.fd = original_fd
                self.lock.filename = original_path
                self.lock.core.set_lock_fd(original_fd, False)
                os.close(new_fd)
            for pid, status in child_statuses:
                self.assertTrue(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0,
                                f"worker {pid} failed: " +
                                next((open(os.path.join(directory, f"error-{i}")).read()
                                      for i, child_pid in enumerate(children)
                                      if child_pid == pid and
                                      os.path.exists(os.path.join(directory, f"error-{i}"))),
                                     "no child traceback"))

    @unittest.skipUnless(hasattr(os, "fork"), "fork is required")
    def test_fork_child_fails_closed_when_lock_reopen_fails(self):
        if self.lock.is_locked:
            self.skipTest("the parent already holds the session build lock")

        original_fd = self.lock.fd
        original_path = self.lock.filename
        original_open = os.open
        parent_pid = os.getpid()
        pid = None
        child_reaped = False
        with tempfile.TemporaryDirectory() as directory:
            lock_path = os.path.join(directory, "jittor.lock")
            result_path = os.path.join(directory, "child-result.json")
            ready_path = os.path.join(directory, "child-failed-closed")
            release_path = os.path.join(directory, "parent-released")
            new_fd = original_open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            self.lock.fd = new_fd
            self.lock.filename = lock_path
            self.lock.core.set_lock_fd(new_fd, False)
            self.lock.lock()

            def fail_child_reopen(path, flags, *args, **kwargs):
                if os.getpid() != parent_pid and os.fspath(path) == lock_path:
                    raise OSError(24, "injected fork-child lock reopen failure")
                return original_open(path, flags, *args, **kwargs)

            try:
                os.open = fail_child_reopen
                pid = os.fork()
                if pid == 0:
                    try:
                        python_error = None
                        try:
                            self.lock.lock()
                        except RuntimeError as error:
                            python_error = str(error)
                        try:
                            self.lock.core.lock_acquire()
                        except BaseException as error:
                            core_result = type(error).__name__ + ": " + str(error)
                        else:
                            core_result = "unexpected success"
                        result = {"python_lock": python_error or "no Python error",
                                  "core_lock_acquire": core_result}
                        with open(result_path, "w") as output:
                            json.dump(result, output)
                        open(ready_path, "w").close()
                        if not self._wait_for_path(release_path):
                            raise TimeoutError("parent did not release the original lock")

                        # Bypass the injected reopen failure with a genuinely
                        # new description, then prove set_lock_fd clears the
                        # fail-closed child state only after rebinding.
                        recovery_fd = original_open(lock_path, os.O_RDWR)
                        self.lock.core.set_lock_fd(recovery_fd, False)
                        self.lock.core.lock_acquire()
                        recovered = self.lock.core.lock_is_held()
                        self.lock.core.lock_release()
                        result["rebound_lock_acquire"] = "succeeded"
                        result["rebound_lock_was_held"] = recovered
                        with open(result_path, "w") as output:
                            json.dump(result, output)
                    except BaseException:
                        result = {"child_error": traceback.format_exc()}
                        with open(result_path, "w") as output:
                            json.dump(result, output)
                        if not os.path.exists(ready_path):
                            open(ready_path, "w").close()
                        os._exit(1)
                    os._exit(0)

                os.open = original_open
                self.assertTrue(self._wait_for_path(ready_path),
                                "child did not report its failed-rebind state")
                result_text = open(result_path).read() if os.path.exists(result_path) else \
                    "child result file missing"
                result = json.loads(result_text)
                self.assertNotIn("child_error", result, result.get("child_error", ""))
                self.assertIn("could not reopen the build lock after fork",
                              result["python_lock"])
                self.assertIn("must be rebound in the fork child",
                              result["core_lock_acquire"])
                self.assertTrue(self.lock.core.lock_is_held(),
                                "child failure changed the parent's in-process lock state")
                probe_fd = original_open(lock_path, os.O_RDWR)
                try:
                    try:
                        jit_lock.fcntl.flock(
                            probe_fd, jit_lock.fcntl.LOCK_EX | jit_lock.fcntl.LOCK_NB)
                    except BlockingIOError:
                        parent_kernel_lock_held = True
                    else:
                        parent_kernel_lock_held = False
                        jit_lock.fcntl.flock(probe_fd, jit_lock.fcntl.LOCK_UN)
                finally:
                    os.close(probe_fd)
                self.assertTrue(parent_kernel_lock_held,
                                "child failure released the parent's kernel lock")

                # The child may acquire on its fresh descriptor only after the
                # parent releases the original OFD.
                self.lock.unlock()
                open(release_path, "w").close()
                _, status = os.waitpid(pid, 0)
                child_reaped = True
                result_text = open(result_path).read() if os.path.exists(result_path) else \
                    "child result file missing"
                self.assertTrue(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0,
                                f"fork child failed with wait status {status}: {result_text}")
                result = json.loads(result_text)
                self.assertNotIn("child_error", result, result.get("child_error", ""))
                self.assertEqual(result.get("rebound_lock_acquire"), "succeeded")
                self.assertTrue(result.get("rebound_lock_was_held"), result)
            finally:
                os.open = original_open
                if pid is not None and not child_reaped:
                    open(release_path, "w").close()
                    if self.lock.is_locked:
                        self.lock.unlock()
                    try:
                        _, status = os.waitpid(pid, 0)
                    except ChildProcessError:
                        pass
                    else:
                        child_reaped = True
                if self.lock.is_locked:
                    self.lock.unlock()
                self.lock.fd = original_fd
                self.lock.filename = original_path
                self.lock.core.set_lock_fd(original_fd, False)
                os.close(new_fd)

    @unittest.skipUnless(hasattr(os, "fork"), "fork is required")
    def test_fork_child_does_not_release_parent_held_lock(self):
        if self.lock.is_locked:
            self.skipTest("the parent already holds the session build lock")

        original_fd = self.lock.fd
        original_path = self.lock.filename
        with tempfile.TemporaryDirectory() as directory:
            lock_path = os.path.join(directory, "jittor.lock")
            ready_path = os.path.join(directory, "ready")
            acquired_path = os.path.join(directory, "acquired")
            error_path = os.path.join(directory, "error")
            new_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            self.lock.fd = new_fd
            self.lock.filename = lock_path
            self.lock.core.set_lock_fd(new_fd, False)
            self.lock.lock()
            pid = os.fork()
            if pid == 0:
                try:
                    import jittor_utils
                    jittor_utils.cc.init_subprocess()
                    jt.jt_init_subprocess()
                    if self.lock.fd == new_fd:
                        raise AssertionError("child kept the inherited fd")
                    open(ready_path, "w").close()
                    self.lock.lock()
                    open(acquired_path, "w").close()
                    self.lock.unlock()
                except BaseException:
                    with open(error_path, "w") as error:
                        error.write(traceback.format_exc())
                    os._exit(1)
                os._exit(0)

            try:
                self.assertTrue(
                    self._wait_for_path(ready_path), "child did not initialize")
                self.assertTrue(self.lock.is_locked)
                self.assertEqual(_lock_state(lock_path), "busy")
                self.assertFalse(os.path.exists(acquired_path),
                                 "child entered while the parent still held the lock")
                self.lock.unlock()
                self.assertTrue(
                    self._wait_for_path(acquired_path), "child did not acquire after release")
            finally:
                if self.lock.is_locked:
                    self.lock.unlock()
                _, status = os.waitpid(pid, 0)
                self.lock.fd = original_fd
                self.lock.filename = original_path
                self.lock.core.set_lock_fd(original_fd, False)
                os.close(new_fd)
            self.assertTrue(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0,
                            "child failed: " +
                            (open(error_path).read() if os.path.exists(error_path)
                             else "no child traceback"))

    @staticmethod
    def _wait_for_path(path, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if os.path.exists(path):
                return True
            time.sleep(0.01)
        return False


@unittest.skipIf(os.name == 'nt', "flock semantics are POSIX only")
class TestBuildLockTimeout(unittest.TestCase):
    """``F_SETLKW`` had no timeout: a wedged holder produced no output, ever."""

    def test_wait_times_out_and_names_the_holder(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "jittor.lock")
            open(path, "w").close()
            # Popen, not a helper runner: the test needs the process alive
            # while it takes the lock, so it still pins PYTHONPATH itself.
            holder = subprocess.Popen([PYTHON, "-c", _HOLDER, path],
                                      env=child_env(),
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE)
            try:
                self.assertEqual(holder.stdout.readline().strip(), b"locked")
                fd = os.open(path, os.O_RDWR)
                old = (jit_lock.lock_timeout, jit_lock.lock_report_after)
                jit_lock.lock_timeout, jit_lock.lock_report_after = 2.0, 0.5
                try:
                    with self.assertRaises(RuntimeError) as caught:
                        jit_lock._acquire(fd, path)
                finally:
                    jit_lock.lock_timeout, jit_lock.lock_report_after = old
                    os.close(fd)
            finally:
                holder.stdin.write(b"\n")
                holder.stdin.close()
                holder.wait(timeout=30)
        message = str(caught.exception)
        self.assertIn("timed out", message)
        # The point of the timeout is the diagnostic, not the exception.
        self.assertIn(str(holder.pid), message)


if __name__ == "__main__":
    unittest.main()
