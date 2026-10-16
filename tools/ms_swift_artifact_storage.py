#!/usr/bin/env python3
"""Single-node Slurm artifact routing, integrity and durable checkpoint publication.

No torch imports, HOME redirection, cache migration or implicit deletion.
The experiment owns the retention plan and calls publish at each stage boundary.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import time

VERSION = "msw-storage-v2"
GIB = 1024 ** 3
LOCAL_FS = {"ext4", "xfs", "btrfs", "tmpfs", "rootfs", "ramfs", "zfs"}
KINDS = {"gradient", "update", "logits", "hidden", "temporary_model"}


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=".json-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, str(path))
        sync_dir(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("artifact path must be a non-empty relative path")
    candidate = Path(root) / relative
    if candidate.resolve() != Path(root).resolve() / relative:
        raise ValueError("symlinked artifact path is forbidden")
    return candidate


def mount(path):
    rows = json.loads(subprocess.check_output([
        "findmnt", "-J", "-T", str(path), "-o", "SOURCE,FSTYPE,TARGET"
    ]).decode())["filesystems"]
    return rows[0]


def ram_budget(reserve):
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    limits = [available]
    # Slurm's requested memory is also a limit even if cgroups are not exposed.
    requested = os.environ.get("SLURM_MEM_PER_NODE")
    if requested:
        limits.append(int(requested) * 1024 ** 2)
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0":
            base = Path("/sys/fs/cgroup") / relative.lstrip("/")
            for ancestor in [base] + list(base.parents):
                cap, used = ancestor / "memory.max", ancestor / "memory.current"
                if cap.is_file() and used.is_file() and cap.read_text().strip() != "max":
                    limits.append(max(0, int(cap.read_text()) - int(used.read_text())))
        elif "memory" in controllers.split(","):
            base = Path("/sys/fs/cgroup/memory") / relative.lstrip("/")
            for ancestor in [base] + list(base.parents):
                cap, used = ancestor / "memory.limit_in_bytes", ancestor / "memory.usage_in_bytes"
                if cap.is_file() and used.is_file():
                    limits.append(max(0, int(cap.read_text()) - int(used.read_text())))
    return {"available_after_training_reserve": max(0, min(limits) - reserve),
            "training_reserve_bytes": reserve, "observed_limits_bytes": limits}


def probe(path, peak, files, ram_reserve=None):
    if peak < 0 or files < 0:
        raise ValueError("negative storage estimate")
    stat = os.statvfs(str(path))
    # peak is the maximum *additional* footprint, including simultaneous
    # staging and all retained outputs declared by the caller.  The source of
    # publish() lives on worker-local scratch, so durable storage needs one
    # staged copy, not a second copy of the declared peak.
    required = peak + 2 * GIB
    required_inodes = files + max(1000, (files + 4) // 5)
    available = stat.f_bavail * stat.f_frsize
    filesystem = mount(path)
    ram = None
    if filesystem["fstype"] in {"tmpfs", "rootfs", "ramfs"}:
        if ram_reserve is None or ram_reserve < 0:
            raise ValueError("memory-backed scratch requires an explicit training RAM reserve")
        ram = ram_budget(ram_reserve)
        available = min(available, ram["available_after_training_reserve"])
    if available < required or stat.f_favail < required_inodes:
        raise OSError("insufficient storage budget: {} bytes, {} inodes required".format(
            required, required_inodes))
    fd, name = tempfile.mkstemp(prefix=".write-probe-", dir=str(path))
    try:
        os.write(fd, b"storage-probe\n")
        os.fsync(fd)
    finally:
        os.close(fd)
        os.unlink(name)
    sync_dir(path)
    return {"available_bytes": available, "free_inodes": stat.f_favail,
            "required_bytes": required, "required_inodes": required_inodes,
            "mount": filesystem, "ram_budget": ram, "fsync_probe": "PASS"}


def validate_plan(plan):
    if not isinstance(plan, dict) or not plan:
        raise ValueError("declare necessary checkpoints before training")
    for name, spec in plan.items():
        if name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("invalid checkpoint name")
        if spec.get("role") not in {"recovery", "validation", "oracle", "failure"}:
            raise ValueError("checkpoint role must be explicit")
        if not spec.get("required") or not spec.get("reason"):
            raise ValueError("checkpoint requires file contract and reason")
        for pattern in spec["required"]:
            if (not Path(pattern).parts or Path(pattern).is_absolute() or
                    ".." in Path(pattern).parts or any(c in pattern for c in "*?[")):
                raise ValueError("invalid checkpoint file contract")


class Storage:
    def __init__(self, evidence):
        self.evidence = Path(evidence).resolve()
        self.manifest = self.evidence / "storage-manifest.json"
        self.data = json.loads(self.manifest.read_text())
        self.work = Path(self.data["work_root"])

    @classmethod
    def create(cls, evidence, plan, scratch_peak, durable_peak, files=1000, ram_reserve=None):
        validate_plan(plan)
        job = os.environ.get("SLURM_JOB_ID", "")
        if not re.fullmatch(r"[0-9]+", job):
            raise RuntimeError("storage initialization requires a Slurm worker")
        if int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) != 1:
            raise RuntimeError("multi-node rank gathering is not yet supported")
        evidence = Path(evidence).absolute()
        evidence.parent.mkdir(parents=True, exist_ok=True)
        if evidence != evidence.resolve():
            raise ValueError("evidence root must be a real path")
        work = Path("/tmp/msw-{}-{}".format(os.getuid(), job))
        if mount("/tmp")["fstype"] not in LOCAL_FS:
            raise RuntimeError("/tmp must be a verified local filesystem")
        if evidence.parent.stat().st_dev == Path("/tmp").stat().st_dev:
            raise RuntimeError("durable evidence and /tmp must be on distinct filesystems")
        # Reserve fresh identities; never reuse historical run keys or another job's files.
        evidence.mkdir(mode=0o700)
        work.mkdir(mode=0o700)
        for folder in ("t", "raw", "models", "trainer", "pycache", "xdg"):
            (work / folder).mkdir(mode=0o700)
        (evidence / "checkpoints").mkdir()
        (evidence / "logs").mkdir()
        data = {"storage_policy_version": VERSION, "state": "preflight",
                "job": job, "node": socket.gethostname(), "uid": os.getuid(),
                "work_root": str(work), "evidence_root": str(evidence),
                "created_at": time.time(), "tmp_retention_days": 10,
                "retention_rule": "site tmpfiles age policy; not guaranteed retention",
                "budgets": {"scratch_peak": scratch_peak, "durable_peak": durable_peak,
                            "files": files, "ram_reserve": ram_reserve},
                "retention_plan": plan, "artifacts": {}, "checkpoints": {}}
        atomic_json(evidence / "storage-manifest.json", data)
        result = cls(evidence)
        try:
            data["preflight"] = {"scratch": probe(work, scratch_peak, files, ram_reserve),
                                 "durable": probe(evidence, durable_peak, files)}
            data["state"] = "ready"
            atomic_json(result.manifest, data)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            data["state"] = "resource-blocked"
            data["failure"] = str(error)
            atomic_json(result.manifest, data)
            raise
        return cls(evidence)

    @contextlib.contextmanager
    def locked(self):
        with open(str(self.evidence / ".storage.lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.data = json.loads(self.manifest.read_text())
            yield
            atomic_json(self.manifest, self.data)

    def assert_worker(self):
        if self.data["state"] != "ready":
            raise RuntimeError("storage is not ready")
        producer = (self.data["node"], self.data["job"])
        consumer = (socket.gethostname(), os.environ.get("SLURM_JOB_ID"))
        if consumer != producer:
            raise RuntimeError("raw artifacts require their producer job and node")

    def raw_path(self, name):
        self.assert_worker()
        path = contained(self.work / "raw", name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def refresh(self):
        """Call before each training stage using conservative declared peak budgets."""
        self.assert_worker()
        budget = self.data["budgets"]
        observation = {
            "scratch": probe(self.work, budget["scratch_peak"], budget["files"], budget["ram_reserve"]),
            "durable": probe(self.evidence, budget["durable_peak"], budget["files"]),
            "at": time.time()}
        with self.locked():
            self.data["latest_preflight"] = observation
        return observation

    def record(self, name, kind, shape=None, dtype=None):
        if kind not in KINDS:
            raise ValueError("unrecognized transient artifact kind")
        path = self.raw_path(name)
        record = {"kind": kind, "path": str(path), "bytes": path.stat().st_size,
                  "sha256": sha256(path), "shape": shape, "dtype": dtype,
                  "node": self.data["node"], "job": self.data["job"],
                  "persistence": "temporary", "expiry_policy_days": 10}
        with self.locked():
            if name in self.data["artifacts"]:
                raise FileExistsError("artifact already recorded; use unique step/rank paths")
            self.data["artifacts"][name] = record
        return record

    def resolve(self, name):
        self.assert_worker()
        data = json.loads(self.manifest.read_text())
        item = data["artifacts"][name]
        path = contained(self.work / "raw", name)
        if not path.is_file():
            raise FileNotFoundError("raw_evidence_expired: " + name)
        if str(path) != item["path"] or sha256(path) != item["sha256"]:
            raise ValueError("raw artifact hash mismatch: " + name)
        return path

    def publish(self, name, source):
        self.assert_worker()
        with self.locked():
            spec = self.data["retention_plan"][name]
            target = self.evidence / "checkpoints" / name
            if target.exists() or name in self.data["checkpoints"]:
                raise FileExistsError("checkpoint publication is immutable")
            source = Path(source).absolute()
            if source != source.resolve() or self.work not in source.parents:
                raise ValueError("publish source must be inside this job's work root")
            for item in source.rglob("*"):
                if item.is_symlink() or (not item.is_dir() and not item.is_file()):
                    raise ValueError("checkpoint contains a link or special file")
            for pattern in spec["required"]:
                if not any(p.is_file() for p in source.glob(pattern)):
                    raise ValueError("missing checkpoint component: " + pattern)
            before = {str(p.relative_to(source)): sha256(p)
                      for p in source.rglob("*") if p.is_file()}
            if not before:
                raise ValueError("empty checkpoint")
            size = sum((source / p).stat().st_size for p in before)
            probe(self.evidence, size, len(before))
            reusable = {}
            for checkpoint_name in self.data["retention_plan"]:
                checkpoint_path = self.evidence / "checkpoints" / checkpoint_name
                if not checkpoint_path.exists():
                    continue
                verified = self.verify_checkpoint(checkpoint_name)
                receipt = json.loads((verified / "storage-receipt.json").read_text())
                for relative, digest in receipt["files"].items():
                    existing = verified / relative
                    reusable.setdefault(digest, existing)
            staging = Path(tempfile.mkdtemp(prefix=".publish-", dir=str(target.parent)))
            # A failed staging tree is retained as failure evidence, never exposed as complete.
            linked = []
            for relative, digest in sorted(before.items()):
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                existing = reusable.get(digest)
                if existing is not None and existing.stat().st_size == (source / relative).stat().st_size:
                    if sha256(existing) != digest:
                        raise ValueError("published checkpoint content hash mismatch")
                    os.chmod(existing, 0o444)
                    os.link(str(existing), str(destination), follow_symlinks=False)
                    linked.append(relative)
                else:
                    shutil.copyfile(str(source / relative), str(destination))
                    with destination.open("rb") as stream:
                        os.fsync(stream.fileno())
                if sha256(destination) != digest:
                    raise ValueError("checkpoint copy hash mismatch")
                os.chmod(destination, 0o444)
            after = {str(p.relative_to(source)): sha256(p)
                     for p in source.rglob("*") if p.is_file()}
            if before != after:
                raise ValueError("checkpoint changed during publication")
            for directory in sorted((p for p in staging.rglob("*") if p.is_dir()),
                                    key=lambda p: len(p.parts), reverse=True):
                sync_dir(directory)
            sync_dir(staging)
            # Receipt travels with the directory: recoverable even if killed before index update.
            receipt = {"role": spec["role"], "reason": spec["reason"], "files": before,
                       "bytes": size, "hardlinked_files": linked,
                       "new_bytes": sum((source / p).stat().st_size for p in before
                                        if p not in linked),
                       "published_at": time.time()}
            if (staging / "storage-receipt.json").exists():
                raise ValueError("reserved checkpoint receipt filename")
            atomic_json(staging / "storage-receipt.json", receipt)
            os.chmod(staging / "storage-receipt.json", 0o444)
            for directory in sorted((p for p in staging.rglob("*") if p.is_dir()),
                                    key=lambda p: len(p.parts), reverse=True):
                os.chmod(directory, 0o555)
            os.chmod(staging, 0o555)
            sync_dir(staging)
            os.rename(str(staging), str(target))
            sync_dir(target.parent)
            self.data["checkpoints"][name] = dict(receipt, path=str(target))
        return target

    def verify_checkpoint(self, name):
        if name not in self.data["retention_plan"]:
            raise ValueError("checkpoint must belong to the declared plan")
        target = contained(self.evidence / "checkpoints", name)
        receipt = json.loads((target / "storage-receipt.json").read_text())
        if any(p.is_symlink() for p in target.rglob("*")):
            raise ValueError("durable checkpoint contains a symlink")
        actual = {str(p.relative_to(target)): sha256(p) for p in target.rglob("*")
                  if p.is_file() and p != target / "storage-receipt.json"}
        if actual != receipt["files"]:
            raise ValueError("durable checkpoint hash mismatch")
        return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    init = commands.add_parser("init")
    init.add_argument("--evidence", required=True)
    init.add_argument("--plan", required=True)
    init.add_argument("--scratch-peak", type=int, required=True)
    init.add_argument("--durable-peak", type=int, required=True)
    init.add_argument("--files", type=int, required=True)
    init.add_argument("--ram-reserve", type=int, required=True)
    for action in ("publish", "verify", "check"):
        sub = commands.add_parser(action)
        sub.add_argument("--evidence", required=True)
        if action != "check":
            sub.add_argument("--name", required=True)
        if action == "publish":
            sub.add_argument("--source", required=True)
    args = parser.parse_args()
    if args.command == "init":
        storage = Storage.create(args.evidence, json.loads(Path(args.plan).read_text()),
                                 args.scratch_peak, args.durable_peak, args.files, args.ram_reserve)
        print(json.dumps({"MSW_EVIDENCE_ROOT": str(storage.evidence),
                          "MSW_WORK_ROOT": str(storage.work),
                          "TMPDIR": str(storage.work / "t")}))
    elif args.command == "publish":
        print(Storage(args.evidence).publish(args.name, args.source))
    elif args.command == "verify":
        print(Storage(args.evidence).verify_checkpoint(args.name))
    elif args.command == "check":
        print(json.dumps(Storage(args.evidence).refresh()))
    else:
        parser.error("a command is required")


if __name__ == "__main__":
    main()
