"""The parity harness must execute on the device it claims to measure.

The runtime scope selects defaults, but NumPy imports have explicit CPU
placement. The runner must move tensors to the requested device and preserve
input gradients; changing the global policy alone cannot establish residency.
Oracle identity, complete gradients and correctness-only acceptance are also
tested here without requiring downstream model runs.
"""

from _helpers import capability as _test_capability

import os
import sys
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import jittor as jt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import child_process  # noqa: E402
from _helpers.runtime_policy import fixture_stack

import _ecosystem_runner  # noqa: E402
import _ecosystem_harness  # noqa: E402


class _StubTorch(object):
    """Stands in for the ``torch`` argument on the Jittor paths."""


class _StubFlags(object):
    use_cuda = 0
    use_acl = 0


def _stub_observation_runtime(flags, acl):
    from jittor._runtime.state import RuntimeContext, RuntimeState
    from jittor._runtime.introspection import EffectivePolicy
    @contextmanager
    def scope(**changes):
        before = {key: getattr(flags, key) for key in changes}
        try:
            for key, value in changes.items():
                setattr(flags, key, value)
            yield
        finally:
            for key, value in before.items():
                setattr(flags, key, value)
    runtime = RuntimeState(RuntimeContext(flags), scope)
    introspection = SimpleNamespace(
        policy=EffectivePolicy(SimpleNamespace(), runtime.context),
        capabilities=SimpleNamespace(backend=lambda name: SimpleNamespace(
            enabled=bool(acl and name == "acl"), failed=False)))
    return runtime, introspection


class _SharedNumpyTensor(object):
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class TestEcosystemDeviceSelection(unittest.TestCase):
    def setUp(self):
        self._policy_stack = fixture_stack(self)

    def test_cpu_request_turns_cuda_off(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1 if _test_capability.check_accelerator('cuda', backend=jt).enabled else 0))
            _ecosystem_runner._select_device(_StubTorch(), "jittor", "cpu", policy_stack=_test_policy_stack)
            self.assertEqual(jt.introspection.policy.runtime.use_cuda, 0)

    def test_cpu_request_is_reported_as_cpu(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1 if _test_capability.check_accelerator('cuda', backend=jt).enabled else 0))
            _ecosystem_runner._select_device(_StubTorch(), "jittor", "cpu", policy_stack=_test_policy_stack)
            self.assertEqual(
                _ecosystem_runner._device_in_use(_StubTorch(), "jittor", "cpu"), "cpu"
            )

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA is unavailable")
    def test_cuda_request_turns_cuda_on_and_is_reported(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=0))
            _ecosystem_runner._select_device(_StubTorch(), "jittor", "cuda", policy_stack=_test_policy_stack)
            self.assertEqual(jt.introspection.policy.runtime.use_cuda, 1)
            self.assertEqual(
                _ecosystem_runner._device_in_use(_StubTorch(), "jittor", "cuda"), "cuda"
            )

    def test_npu_request_requires_acl_and_is_reported_separately(self):
        flags = _StubFlags()
        runtime, observation = _stub_observation_runtime(flags, True)
        with mock.patch.object(jt, "runtime", runtime):
            with mock.patch.object(jt, "introspection", observation):
                _ecosystem_runner._select_device(_StubTorch(), "jittor", "npu", policy_stack=self._policy_stack)
                self.assertEqual(flags.use_cuda, 1)
                self.assertEqual(flags.use_acl, 1)
                self.assertEqual(
                    _ecosystem_runner._device_in_use(
                        _StubTorch(), "jittor", "npu"
                    ),
                    "npu",
                )

    def test_npu_request_fails_without_acl(self):
        runtime, observation = _stub_observation_runtime(_StubFlags(), False)
        with mock.patch.object(jt, "runtime", runtime), mock.patch.object(jt, "introspection", observation):
            with self.assertRaisesRegex(SystemExit, "ACL is unavailable"):
                _ecosystem_runner._select_device(_StubTorch(), "jittor", "npu", policy_stack=self._policy_stack)

    def test_jittor_mover_honors_explicit_tensor_placement(self):
        move = _ecosystem_runner._select_device(
            _StubTorch(), "jittor", "cpu", policy_stack=self._policy_stack,
        )
        source = mock.Mock()
        moved = object()
        source.to.return_value = moved
        self.assertIs(move(source), moved)
        source.to.assert_called_once_with("cpu")

    def _assert_numpy_input_gradient_on_device(self, device):
        from contextlib import ExitStack
        import torch

        with ExitStack() as stack:
            move = _ecosystem_runner._select_device(
                torch, "jittor", device, policy_stack=stack,
            )
            stack.enter_context(jt.runtime.scope(backend_fallback="error"))
            before = jt.core.backend_fallback_count()
            inputs = _ecosystem_runner._make_inputs(
                torch, {"x": ("float32", (2,), None)}, 1, move,
            )
            value = inputs["x"]
            self.assertTrue(value.requires_grad)
            self.assertTrue(value.is_leaf)
            self.assertEqual(value.placement_backend == 0, device == "cpu")
            loss = (value * value).sum()
            loss.backward()
            self.assertIsNotNone(value.grad)
            jt.sync([loss, value.grad], device_sync=device == "cuda")
            location = "device" if device == "cuda" else "cpu"
            self.assertEqual(loss.location(), location)
            self.assertEqual(value.grad.location(), location)
            expected = 2 * np.random.RandomState(1).randn(2).astype("float32")
            np.testing.assert_allclose(value.grad.cpu().numpy(), expected)
            self.assertEqual(jt.core.backend_fallback_count(), before)

    def test_cpu_mover_preserves_numpy_input_gradient(self):
        self._assert_numpy_input_gradient_on_device("cpu")

    @unittest.skipUnless(_test_capability.check_accelerator("cuda", backend=jt).enabled, "CUDA is unavailable")
    def test_cuda_mover_preserves_numpy_input_gradient(self):
        self._assert_numpy_input_gradient_on_device("cuda")

    def test_shared_package_site_is_inserted_without_duplicates(self):
        original = list(sys.path)

        def restore_path():
            sys.path[:] = original

        self.addCleanup(restore_path)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"JITTOR_ECOSYSTEM_PACKAGE_SITE": directory},
            ):
                sys.path.extend([directory, directory])
                actual = _ecosystem_runner._activate_package_site()
                self.assertEqual(actual, str(Path(directory).resolve()))
                self.assertEqual(sys.path[0], actual)
                self.assertEqual(sys.path.count(actual), 1)

    def test_harness_selects_independent_reference_package_site(self):
        with mock.patch.object(
            _ecosystem_harness, "PACKAGE_SITE", "/packages/python39"
        ):
            with mock.patch.object(
                _ecosystem_harness,
                "REFERENCE_PACKAGE_SITE",
                "/packages/python310",
            ):
                with mock.patch.object(
                    _ecosystem_harness, "REFERENCE_SHARES_PACKAGE_SITE", False
                ):
                    self.assertEqual(
                        _ecosystem_harness._runner_package_site(child_process.PYTHON),
                        "/packages/python39",
                    )
                    self.assertEqual(
                        _ecosystem_harness._runner_package_site(
                            "/oracle/bin/python"
                        ),
                        "/packages/python310",
                    )

    def test_harness_shares_package_site_for_compatible_abis(self):
        with mock.patch.object(
            _ecosystem_harness, "PACKAGE_SITE", "/packages/shared"
        ):
            with mock.patch.object(
                _ecosystem_harness, "REFERENCE_PACKAGE_SITE", ""
            ):
                with mock.patch.object(
                    _ecosystem_harness, "REFERENCE_SHARES_PACKAGE_SITE", True
                ):
                    self.assertEqual(
                        _ecosystem_harness._runner_package_site(
                            "/oracle/bin/python"
                        ),
                        "/packages/shared",
                    )

    def test_correctness_snapshot_does_not_alias_runtime_storage(self):
        storage = np.array([1.0, 2.0], dtype="float32")
        snapshot = _ecosystem_runner._numpy_snapshot(_SharedNumpyTensor(storage))
        storage[:] = -1.0
        np.testing.assert_array_equal(snapshot, np.array([1.0, 2.0], dtype="float32"))


class TestEcosystemStrictTimmContracts(unittest.TestCase):
    def test_required_torchvision_import_failure_cannot_activate_facade(self):
        reference = SimpleNamespace(_C=object())
        with mock.patch.dict(sys.modules, {"torch": reference, "torchvision": None}), \
                mock.patch.dict(os.environ, {}), \
                mock.patch.object(_ecosystem_runner, "_activate_package_site") as activate:
            with self.assertRaises(ImportError):
                _ecosystem_runner._import_torch("torch", require_torchvision=True)
            activate.assert_not_called()

    def test_required_torchvision_must_come_from_installed_distribution(self):
        distribution = SimpleNamespace(
            locate_file=lambda path: Path("/oracle/site-packages") / path,
        )
        with mock.patch.object(_ecosystem_runner._distribution_metadata,
                               "distribution", return_value=distribution):
            with self.assertRaisesRegex(RuntimeError, "torchvision oracle imported"):
                _ecosystem_runner._validate_reference_torchvision(
                    SimpleNamespace(__file__="/shim/torchvision/__init__.py"),
                )
            _ecosystem_runner._validate_reference_torchvision(
                SimpleNamespace(__file__="/oracle/site-packages/torchvision/__init__.py"),
            )

    def test_each_required_parameter_or_input_gradient_must_exist(self):
        value = SimpleNamespace(requires_grad=True, grad=None)
        for parameter in (True, False):
            with self.subTest(parameter=parameter):
                model = SimpleNamespace(named_parameters=lambda: [("weight", value)] if parameter else [])
                inputs = {} if parameter else {"x": value}
                expected = "grad::weight" if parameter else "ingrad::x"
                with self.assertRaisesRegex(RuntimeError, expected):
                    _ecosystem_runner._gradient_snapshots(model, inputs, require_complete=True)

    def test_required_gradients_must_be_finite(self):
        value = SimpleNamespace(
            requires_grad=True,
            grad=_SharedNumpyTensor(np.array([np.nan], dtype="float32")),
        )
        model = SimpleNamespace(named_parameters=lambda: [("weight", value)])
        with self.assertRaisesRegex(RuntimeError, "non-finite gradient: grad::weight"):
            _ecosystem_runner._gradient_snapshots(model, {}, require_complete=True)

    def test_frozen_parameters_and_existing_optional_policy_are_preserved(self):
        value = SimpleNamespace(requires_grad=False, grad=None)
        model = SimpleNamespace(named_parameters=lambda: [("weight", value)])
        self.assertEqual(_ecosystem_runner._gradient_snapshots(model, {}, require_complete=True), {})
        value.requires_grad = True
        self.assertEqual(_ecosystem_runner._gradient_snapshots(model, {}), {})


class TestEcosystemCorrectnessOnly(unittest.TestCase):
    def _compare_reports(self, correctness_only, fault=None):
        import io
        from contextlib import redirect_stdout

        comparison = _ecosystem_harness.EcosystemComparison()
        comparison.device = "cuda"

        def run(python, runtime, case, output, **kwargs):
            arrays = {
                "__output__": np.array([1.], dtype="float32"),
                "grad::weight": np.array([2.], dtype="float32"),
                "ingrad::x": np.array([3.], dtype="float32"),
            }
            report = {
                "device": "cuda", "dependencies": {}, "tf32": {"matmul": False},
                "fallback_count": 0, "fallback_policy": "error", "seconds": 1.,
                "runtime_conditions": {
                    "runtime_threads": 2 if runtime == "torch" else 64,
                    "affinity": [0, 1], "thread_env": {"OMP_NUM_THREADS": "2"},
                    "precision": {"matmul": False},
                },
            }
            if runtime == "torch":
                np.savez(Path(output).with_suffix(".weights.npz"), weight=np.ones(1))
            elif fault == "device":
                report["device"] = "cpu"
            elif fault == "precision":
                report["tf32"]["matmul"] = True
            elif fault == "input_gradient":
                arrays.pop("ingrad::x")
            elif fault == "affinity":
                report["runtime_conditions"]["affinity"] = [0]
            elif fault == "thread_env":
                report["runtime_conditions"]["thread_env"]["OMP_NUM_THREADS"] = "1"
            elif fault == "fallback":
                report["fallback_count"] = 1
            np.savez(output, **arrays)
            return report, ""

        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                _ecosystem_harness, "_distributions_available", return_value=True))
            stack.enter_context(mock.patch.object(
                _ecosystem_harness, "_runner_package_site", return_value=""))
            stack.enter_context(mock.patch.object(
                _ecosystem_harness, "_run", side_effect=run))
            stack.enter_context(mock.patch.object(_ecosystem_harness, "SPEED_RATIO", "0.01"))
            stack.enter_context(redirect_stdout(output))
            comparison._compare("timm_resnet18_tiny", correctness_only=correctness_only)
        return output.getvalue()

    def test_correctness_only_reports_unknown_threads_and_no_performance(self):
        import json

        output = self._compare_reports(True)
        self.assertNotIn("[speed/", output)
        report = json.loads(output.split("CORRECTNESS_RESULT ", 1)[1])
        self.assertFalse(report["performance_validated"])
        self.assertFalse(report["jittor_thread_control_available"])
        self.assertIsNone(report["jittor_effective_runtime_threads"])
        self.assertEqual(report["reported_runtime_threads"], {"torch": 2, "jittor": 64})

    def test_default_mode_still_rejects_different_thread_counts(self):
        with self.assertRaisesRegex(AssertionError, "different thread counts"):
            self._compare_reports(False)

    def test_correctness_only_keeps_device_precision_gradient_and_runtime_guards(self):
        for fault in ("device", "precision", "input_gradient", "affinity", "thread_env", "fallback"):
            with self.subTest(fault=fault):
                with self.assertRaises(AssertionError):
                    self._compare_reports(True, fault)


class TestEcosystemTensorResidency(unittest.TestCase):
    def _check_residency(self, fault=None, wrong_backend="cpu"):
        def tensor():
            value = SimpleNamespace(backend="cuda", index=0, requires_grad=False)
            value.location = lambda: "cpu" if value.backend == "cpu" else "device"
            return value

        values = {name: tensor() for name in (
            "output", "param::weight", "buffer::count", "input::x",
            "grad::weight", "ingrad::x",
        )}
        values["param::weight"].requires_grad = True
        values["param::weight"].grad = values["grad::weight"]
        values["input::x"].requires_grad = True
        values["input::x"].grad = values["ingrad::x"]
        if fault:
            values[fault].backend = wrong_backend
            values[fault].index = -1 if wrong_backend == "cpu" else 0
        model = SimpleNamespace(
            named_parameters=lambda: [("weight", values["param::weight"])],
            named_buffers=lambda: [("count", values["buffer::count"])],
        )
        native = SimpleNamespace(
            sync=mock.Mock(),
            core=SimpleNamespace(
                current_device=lambda: 0,
                dispatch_context=lambda args: (args[0].backend, args[0].index),
            ),
        )
        with mock.patch.dict(sys.modules, {"jittor": native}):
            return _ecosystem_runner._timm_tensor_residency(
                _StubTorch(), model, {"x": values["input::x"]}, values["output"],
                "jittor", "cuda",
            )

    def test_global_cuda_cannot_hide_any_cpu_tensor(self):
        for name in ("output", "param::weight", "buffer::count", "input::x",
                     "grad::weight", "ingrad::x"):
            with self.subTest(tensor=name):
                with self.assertRaisesRegex(RuntimeError, name):
                    self._check_residency(name)

    def test_other_accelerator_cannot_masquerade_as_cuda(self):
        with self.assertRaisesRegex(RuntimeError, "expected cuda:0"):
            self._check_residency("output", wrong_backend="acl")

    def test_residency_report_covers_every_tensor_group(self):
        report = self._check_residency()
        self.assertTrue(report["all_on_requested_device"])
        self.assertEqual(report["checked_tensors"], 6)
        self.assertEqual(report["backend"], "cuda")
        self.assertEqual(report["groups"], {
            "output": 1, "param": 1, "buffer": 1, "input": 1,
            "grad": 1, "ingrad": 1,
        })


if __name__ == "__main__":
    unittest.main()
