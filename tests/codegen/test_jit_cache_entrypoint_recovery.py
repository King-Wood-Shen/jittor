"""A matching cache key must not bless a JIT library without its entrypoint."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from _helpers.child_process import run_child_script


_BUILD_OP = r"""
import glob
import os
import jittor as jt

jt.flags.use_cuda = 0
got = (jt.array([10, 20], dtype="int32") - 1).numpy().tolist()
assert got == [9, 19], got
products = glob.glob(jt.compiler.cache_path + "/jit/*.so")
assert products, jt.compiler.cache_path
print("JIT_PRODUCT=" + max(products, key=os.path.getmtime))
print("RECOVERY_RESULT=" + repr(got))
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="this recovery contract uses ELF shared objects")
def test_bad_jit_entrypoint_invalidates_a_matching_cache_key(tmp_path):
    home = tmp_path / "jittor-home"
    home.mkdir()
    env = {"JITTOR_HOME": home, "cache_name": "entrypoint-recovery"}

    first = run_child_script(_BUILD_OP, env=env, timeout=900, text=True,
                             merge_stderr=True, directory=tmp_path,
                             name="build-jit-op")
    assert first.returncode == 0, first.stdout
    product_line = next(line for line in first.stdout.splitlines()
                        if line.startswith("JIT_PRODUCT="))
    product = Path(product_line.partition("=")[2])
    key = Path(str(product) + ".key")
    recorded_key = key.read_bytes()
    assert recorded_key

    invalid_source = tmp_path / "missing_entrypoint.cc"
    invalid_source.write_text(
        'extern "C" int unrelated_entrypoint() { return 1; }\n',
        encoding="utf-8",
    )
    invalid_product = tmp_path / "missing_entrypoint.so"
    subprocess.run(
        [os.environ.get("CXX", "g++"), "-shared", "-fPIC",
         str(invalid_source), "-o", str(invalid_product)],
        check=True,
    )
    os.replace(invalid_product, product)
    assert key.read_bytes() == recorded_key
    invalid_inode = product.stat().st_ino

    second = run_child_script(_BUILD_OP, env=env, timeout=900, text=True,
                              merge_stderr=True, directory=tmp_path,
                              name="recover-jit-op")
    assert second.returncode == 0, second.stdout
    assert "RECOVERY_RESULT=[9, 19]" in second.stdout
    assert product.stat().st_ino != invalid_inode, \
        "the matching key caused the invalid shared object to be reused"
    assert key.read_bytes() == recorded_key
