"""Storage-chain tests: run directly with unittest on a Slurm worker, no JIT."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

class TestArtifactStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global storage
        spec = importlib.util.spec_from_file_location(
            "msw_storage", Path(__file__).resolve().parents[2] / "tools/ms_swift_artifact_storage.py")
        storage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(storage)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        work = self.root / "work"
        work.mkdir()
        (work / "raw").mkdir()
        (self.root / "checkpoints").mkdir()
        self.plan = {"restore": {"role": "recovery", "reason": "new-process restore",
                                "required": ["model.bin", "optimizer.json", "rng_state_0.json", "rng_state_1.json"]},
                     "restore-copy": {"role": "validation", "reason": "same immutable state",
                                      "required": ["model.bin", "optimizer.json", "rng_state_0.json", "rng_state_1.json"]}}
        storage.atomic_json(self.root / "storage-manifest.json", {
            "work_root": str(work), "state": "ready", "node": socket.gethostname(),
            "job": "123", "retention_plan": self.plan, "artifacts": {}, "checkpoints": {}})
        self.s = storage.Storage(self.root)
        self.env = patch.dict(os.environ, {"SLURM_JOB_ID": "123"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def checkpoint(self):
        cp = self.s.work / "trainer" / "checkpoint-2"
        cp.mkdir(parents=True)
        for name in self.plan["restore"]["required"]:
            (cp / name).write_bytes(("state-fixture:" + name).encode("utf-8"))
        return cp

    def test_identical_published_files_share_durable_storage(self):
        first = self.checkpoint()
        with patch.object(storage, "probe", return_value={}):
            first_target = self.s.publish("restore", first)
            second = self.s.work / "trainer" / "checkpoint-3"
            second.mkdir(parents=True)
            for name in self.plan["restore-copy"]["required"]:
                (second / name).write_bytes(("state-fixture:" + name).encode("utf-8"))
            second_target = self.s.publish("restore-copy", second)
        receipt = json.loads((second_target / "storage-receipt.json").read_text())
        self.assertEqual((first_target / "model.bin").stat().st_ino,
                         (second_target / "model.bin").stat().st_ino,
                         "hardlink receipt: {!r}".format(receipt))
        self.assertEqual(sorted(receipt["hardlinked_files"]), sorted(self.plan["restore-copy"]["required"]))
        self.assertEqual(receipt["new_bytes"], 0)
        self.assertEqual((second_target / "model.bin").stat().st_mode & 0o222, 0)
        self.assertEqual(self.s.verify_checkpoint("restore-copy"), second_target)

    def test_hash_mismatch_is_rejected(self):
        path = self.s.raw_path("rank0/step1.npy")
        path.write_bytes(b"initial")
        self.s.record("rank0/step1.npy", "gradient", [1], "float32")
        self.assertEqual(self.s.resolve("rank0/step1.npy"), path)
        path.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.s.resolve("rank0/step1.npy")

    def test_expired_and_wrong_node_are_not_silent(self):
        path = self.s.raw_path("update.npy")
        path.write_bytes(b"step")
        self.s.record("update.npy", "update")
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "raw_evidence_expired"):
            self.s.resolve("update.npy")
        with patch.object(storage.socket, "gethostname", return_value="other-node"):
            with self.assertRaisesRegex(RuntimeError, "producer"):
                self.s.raw_path("new.npy")

    def test_paths_and_duplicate_raw_records_are_rejected(self):
        with self.assertRaises(ValueError):
            self.s.raw_path("../outside")
        (self.s.work / "raw" / "link").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.s.raw_path("link/file")
        path = self.s.raw_path("unique.npy")
        path.write_bytes(b"x")
        self.s.record("unique.npy", "gradient")
        with self.assertRaises(FileExistsError):
            self.s.record("unique.npy", "gradient")

    def test_publish_survives_temporary_loss_and_is_immutable(self):
        cp = self.checkpoint()
        with patch.object(storage, "probe", return_value={}):
            target = self.s.publish("restore", cp)
        self.assertEqual(self.s.verify_checkpoint("restore"), target)
        import shutil
        shutil.rmtree(str(cp))
        self.assertEqual(self.s.verify_checkpoint("restore"), target)
        with self.assertRaises(FileExistsError):
            self.s.publish("restore", cp)
        (target / "model.bin").chmod(0o644)
        (target / "model.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.s.verify_checkpoint("restore")

    def test_missing_rank_state_and_source_links_block_publication(self):
        cp = self.checkpoint()
        (cp / "rng_state_1.json").unlink()
        with self.assertRaisesRegex(ValueError, "rng_state_1"):
            self.s.publish("restore", cp)
        (cp / "rng_state_1.json").symlink_to(cp / "rng_state_0.json")
        with self.assertRaisesRegex(ValueError, "link"):
            self.s.publish("restore", cp)
        self.assertFalse((self.root / "checkpoints" / "restore").exists())

    def test_capacity_failure_and_multinode_block_before_training(self):
        with self.assertRaisesRegex(OSError, "insufficient"):
            storage.probe(self.root, 10 ** 30, 1, 0)
        with patch.dict(os.environ, {"SLURM_JOB_NUM_NODES": "2"}):
            with self.assertRaisesRegex(RuntimeError, "multi-node"):
                storage.Storage.create(self.root / "future", self.plan, 0, 0)
        self.assertFalse((self.root / "future").exists())

    def test_capacity_budget_counts_one_durable_staging_copy_plus_reserve(self):
        from types import SimpleNamespace
        peak = 4 * storage.GIB
        free = peak + 2 * storage.GIB
        stat = SimpleNamespace(f_bavail=free, f_frsize=1, f_favail=10000)
        with patch.object(storage.os, "statvfs", return_value=stat), \
                patch.object(storage, "mount", return_value={"fstype": "ext4"}):
            result = storage.probe(self.root, peak, 1)
            self.assertEqual(result["required_bytes"], free)
            too_little = SimpleNamespace(f_bavail=free - 1, f_frsize=1, f_favail=10000)
            with patch.object(storage.os, "statvfs", return_value=too_little):
                with self.assertRaisesRegex(OSError, "insufficient"):
                    storage.probe(self.root, peak, 1)

    def test_plan_needs_reason_and_component_contract(self):
        with self.assertRaises(ValueError):
            storage.validate_plan({"bad": {"role": "validation"}})
        for name, required in (("..", ["model.bin"]), ("safe", ["model-*.bin"])):
            with self.assertRaises(ValueError):
                storage.validate_plan({name: {"role": "recovery", "reason": "restore", "required": required}})

    def test_ram_backed_scratch_counts_job_memory(self):
        with patch.object(storage, "mount", return_value={"fstype": "rootfs"}):
            with self.assertRaisesRegex(ValueError, "RAM reserve"):
                storage.probe(self.root, 1, 1)
            with patch.object(storage, "ram_budget", return_value={"available_after_training_reserve": 1}):
                with self.assertRaisesRegex(OSError, "insufficient"):
                    storage.probe(self.root, 1, 1, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
