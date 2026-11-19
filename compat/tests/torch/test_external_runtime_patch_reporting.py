"""Reporting regressions for external Torch runtime patches."""

import sys
import types
import unittest
from unittest import mock

from jittor.compat import external_backend, integrations, module_patcher


class TestPatchReporting(unittest.TestCase):
    def test_deprecated_vllm_module_is_not_imported_by_root_runtime(self):
        patch_report = module_patcher.PatchReport((), True)
        backend_report = ()
        triton_compat = types.ModuleType("jittor.compat.triton")
        with mock.patch.object(
            module_patcher, "install_module_patches", return_value=patch_report
        ), mock.patch.object(
            external_backend, "load_external_backend_entry_points", return_value=backend_report
        ), mock.patch.object(integrations, "swallowed") as swallowed_mock, mock.patch.dict(
            sys.modules, {"jittor.compat.triton": triton_compat}
        ):
            report = integrations.apply_external_runtime_patches(transaction=object())

        self.assertNotIn("vllm_shim", report)
        self.assertFalse(
            any("vllm" in call.args[0] for call in swallowed_mock.call_args_list)
        )

    def test_root_runtime_report_is_readable_and_repeatable(self):
        patch_report = module_patcher.PatchReport(
            (
                module_patcher.PatchResult(
                    "entry_point", "broken", "provider:broken", "failed", "first"
                ),
                module_patcher.PatchResult("entry_point", "working", "provider:working", "loaded"),
            ),
            True,
        )
        backend_report = (
            external_backend.BackendEntryPointResult(
                "broken-backend", "provider:backend", "failed", "second"
            ),
            external_backend.BackendEntryPointResult(
                "working-backend", "provider:working", "loaded"
            ),
        )
        triton_compat = types.ModuleType("jittor.compat.triton")
        with mock.patch.object(
            module_patcher, "install_module_patches", return_value=patch_report
        ) as patch_mock, mock.patch.object(
            external_backend,
            "load_external_backend_entry_points",
            return_value=backend_report,
        ) as backend_mock, mock.patch.dict(
            sys.modules, {"jittor.compat.triton": triton_compat}
        ):
            first = integrations.apply_external_runtime_patches(transaction=object())
            second = integrations.apply_external_runtime_patches(transaction=object())

        self.assertEqual(patch_mock.call_count, 2)
        self.assertEqual(backend_mock.call_count, 2)
        self.assertTrue(first["triton_shim"]["ok"])
        self.assertFalse(first["module_patches"]["ok"])
        self.assertEqual(first["module_patches"]["results"][0]["detail"], "first")
        self.assertFalse(first["external_backends"]["ok"])
        self.assertEqual(first["external_backends"]["results"][0]["detail"], "second")
        self.assertEqual(first, second)
        self.assertIs(integrations.apply_external_runtime_patches.last_report, second)
