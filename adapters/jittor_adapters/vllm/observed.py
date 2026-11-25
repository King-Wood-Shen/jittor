"""Explicit opt-in for the observed Qwen3 route of vLLM 0.11.0."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile


_VERSION = "0.11.0"
_GRAPH_SHA256 = "a1d3e7078beb6d6c8a98be00dbc3e387799ff609213c6a3c4e283f2cd861c1e2"


def _verify_overlay(package_root):
    manifest = Path(__file__).with_name(f"vllm-{_VERSION}.sha256")
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        if hashlib.sha256((package_root / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Unexpected vLLM adapter source: {relative}")


def prepare(wheel, destination):
    """Materialize a versioned adapter into a new directory from a pinned wheel."""
    if not wheel.name.startswith(f"vllm-{_VERSION}-"):
        raise ValueError(f"Expected a vLLM {_VERSION} wheel")
    with wheel.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != "3861c75ff2b12e24f6d179ff5c084d791b42ded8675d76c8706697c79f68cd62":
        raise ValueError("vLLM wheel digest mismatch")
    destination.mkdir()
    with ZipFile(wheel) as archive:
        files = [name for name in archive.namelist()
                 if name.startswith(("vllm/", f"vllm-{_VERSION}.dist-info/"))]
        archive.extractall(destination, files)
    patch = Path(__file__).with_name(f"vllm-{_VERSION}.patch")
    with patch.open("rb") as stream:
        subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1"],
                       cwd=destination, stdin=stream, check=True)
    _verify_overlay(destination)


def activate(graph_file, out):
    """Arm the observed graph before any vLLM model is constructed."""
    import jittor
    import torch
    import vllm

    if (torch.owner is not jittor or vllm.__version__ != _VERSION
            or os.environ.get("JT_VLLM_OBSERVED_COMPILE") != "1"):
        raise RuntimeError(f"Expected the opt-in Jittor adapter for vLLM {_VERSION}")
    _verify_overlay(Path(vllm.__file__).resolve().parents[1])
    if hashlib.sha256(graph_file.read_bytes()).hexdigest() != _GRAPH_SHA256:
        raise ValueError("Unexpected Qwen3 decoder graph")
    return jittor.compat.bind_observed_qwen3(graph_file, out)


if __name__ == "__main__":
    prepare(*map(Path, sys.argv[1:]))
