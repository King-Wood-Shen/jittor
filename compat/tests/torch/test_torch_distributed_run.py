"""Public torchrun arguments delegate accelerator placement to jtrun."""

import importlib.util
import os
import unittest
from unittest import mock

import torch


class TestDistributedRun(unittest.TestCase):
    def test_swift_arguments_and_hccl_rank_masks(self):
        self.assertIsNotNone(importlib.util.find_spec("torch.distributed.run").loader)
        from torch.distributed.run import _native_arguments
        from jittor.distributed.launch import _parse_args, _rank_environment

        self.assertIs(torch.distributed.run._native_arguments, _native_arguments)
        args = _parse_args(
            _native_arguments(
                [
                    "--nproc_per_node",
                    "2",
                    "--master_port",
                    "29601",
                    "--nnodes",
                    "1",
                    "--node_rank",
                    "0",
                    "--master_addr",
                    "127.0.0.1",
                    "train.py",
                    "--max_steps",
                    "4",
                    "--module",
                    "model-option",
                ]
            )
        )
        self.assertEqual(args.backend, "auto")
        self.assertEqual(args.cmd[1:], ["train.py", "--max_steps", "4", "--module", "model-option"])
        self.assertEqual(args.master_port, 29601)
        with mock.patch.dict(os.environ, {"ASCEND_VISIBLE_DEVICES": "3,5"}):
            env, local_rank, device = _rank_environment(
                args, 1, "hccl", ["3", "5"], "/tmp/rootinfo", "127.0.0.1", 29601
            )
        self.assertEqual(device, "5")
        self.assertEqual(local_rank, 0)
        self.assertEqual(env["RANK"], "1")
        self.assertEqual(env["WORLD_SIZE"], "2")
        self.assertEqual(env["LOCAL_RANK"], "0")
        for name in (
            "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_VISIBLE_DEVICES",
            "MS_SWIFT_VISIBLE_DEVICES",
        ):
            self.assertEqual(env[name], "5")

    def test_module_standalone_and_rejected_elastic_options(self):
        from torch.distributed.run import _native_arguments
        from jittor.distributed.launch import _parse_args

        args = _parse_args(_native_arguments(["--standalone", "--module", "train", "--flag"]))
        self.assertEqual(args.cmd[1:], ["-m", "train", "--flag"])
        self.assertEqual(args.master_port, 0)
        for unsupported in (["--nnodes", "2"], ["--max-restarts", "1"], ["--nproc-per-node", "0"]):
            with self.subTest(unsupported=unsupported), self.assertRaises(SystemExit):
                _native_arguments(unsupported + ["train.py"])
