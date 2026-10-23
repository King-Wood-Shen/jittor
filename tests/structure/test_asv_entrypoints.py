"""Unavailable benchmark parameters must not hide broken installations or kernels."""

import ast
import itertools
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

from benchmarks.asv import _shared
from benchmarks.asv.accelerate_training import _configure_device
from _helpers.child_process import run_python_child


ROOT = Path(__file__).resolve().parents[2]


class TestAsvBackendSelection(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="asv-backend-")
        self.addCleanup(self.directory.cleanup)
        self.environment = mock.patch.dict(
            os.environ,
            {
                "JITTOR_HOME": str(Path(self.directory.name) / "asv"),
                "cache_name": "asv-selection-test",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_only_absent_optional_torch_is_a_skip(self):
        with mock.patch.object(
            _shared.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("No module named torch", name="torch"),
        ):
            with self.assertRaisesRegex(NotImplementedError, "not installed"):
                _shared.load_backend("torch", "cpu")
        for error in (
            ModuleNotFoundError("missing binary dependency", name="torch._C"),
            OSError("broken shared library"),
        ):
            with self.subTest(error=error), mock.patch.object(
                _shared.importlib, "import_module", side_effect=error
            ):
                with self.assertRaises(type(error)) as caught:
                    _shared.load_backend("torch", "cpu")
                self.assertIs(caught.exception, error)

    def test_cuda_compile_and_execution_failures_are_not_skips(self):
        error = RuntimeError("kernel compilation failed")
        backend = SimpleNamespace(
            flags=SimpleNamespace(use_cuda=0),
            compiler=SimpleNamespace(has_cuda=True),
            ones=mock.Mock(side_effect=error),
        )
        with mock.patch.object(_shared.importlib, "import_module", return_value=backend):
            with self.assertRaises(RuntimeError) as caught:
                _shared.load_backend("jittor", "cuda")
            self.assertIs(caught.exception, error)
            backend.compiler.has_cuda = False
            with self.assertRaisesRegex(NotImplementedError, "without CUDA"):
                _shared.load_backend("jittor", "cuda")

    def test_cpu_explicitly_disables_an_inherited_accelerator(self):
        backend = SimpleNamespace(flags=SimpleNamespace(use_cuda=1))
        with mock.patch.object(_shared.importlib, "import_module", return_value=backend):
            self.assertIs(_shared.load_backend("jittor", "cpu"), backend)
        self.assertEqual(backend.flags.use_cuda, 0)

    def test_oracle_requires_binary_torch_and_rejects_a_facade(self):
        backend = SimpleNamespace(__file__="/site/torch/__init__.py")
        with mock.patch.object(_shared.importlib, "import_module", return_value=backend):
            with self.assertRaisesRegex(RuntimeError, "binary"):
                _shared.load_backend("torch", "cpu")
            backend._C = SimpleNamespace(__file__="/site/torch/_C.so")
            self.assertIs(_shared.load_backend("torch", "cpu"), backend)
            backend.__file__ = "/lab/jittor-lab/_state/oracle/site/torch/__init__.py"
            self.assertIs(_shared.load_backend("torch", "cpu"), backend)
            backend._torch_compat_install_context = object()
            with self.assertRaisesRegex(RuntimeError, "shim"):
                _shared.load_backend("torch", "cpu")

    def test_unknown_devices_fail_before_importing_a_framework(self):
        with mock.patch.object(_shared.importlib, "import_module") as importer:
            with self.assertRaisesRegex(ValueError, "device"):
                _shared.load_backend("jittor", "cdua")
            with self.assertRaisesRegex(ValueError, "backend"):
                _shared.load_backend("other", "cpu")
            importer.assert_not_called()


class TestAsvEntryPoints(unittest.TestCase):
    def test_accelerate_cpu_only_build_never_reads_cuda_only_flags(self):
        class CpuFlags:
            __slots__ = ("use_cuda",)

        backend = SimpleNamespace(flags=CpuFlags(), compiler=SimpleNamespace(has_cuda=False))
        _configure_device(backend, "cpu")
        self.assertEqual(backend.flags.use_cuda, 0)
        with self.assertRaisesRegex(NotImplementedError, "CUDA build"):
            _configure_device(backend, "cuda")
        with self.assertRaisesRegex(ValueError, "device"):
            _configure_device(backend, "npu")

    def test_source_config_is_executable_without_nox_path_rewriting(self):
        environment = dict(os.environ)
        environment.pop("ASV_CONF_DIR", None)
        result = run_python_child(
            [
                "-m",
                "asv",
                "check",
                "--config",
                "benchmarks/asv/asv.conf.json",
                "--python=same",
            ],
            cwd=str(ROOT),
            env=environment,
            inherit=False,
            timeout=120,
            merge_stderr=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("No problems found", result.stdout)
        directory = ROOT / "benchmarks" / "asv"
        config = json.loads((directory / "asv.conf.json").read_text())
        for key in ("env_dir", "results_dir", "html_dir"):
            target = (directory / config[key]).resolve()
            self.assertNotIn(ROOT, target.parents, key)

    def test_opt_in_accelerate_default_does_not_select_unrelated_benchmarks(self):
        tree = ast.parse((ROOT / "noxfile.py").read_text())
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_asv_run_arguments"
        )
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "noxfile.py", "exec"), namespace)
        select = namespace["_asv_run_arguments"]
        pattern = r"^accelerate_training\."
        self.assertEqual(select(("--quick",), pattern), ("--bench", pattern, "--quick"))
        for arguments in (("-b", "convolution"), ("--bench=convolution",), ("-bconvolution",)):
            self.assertEqual(select(arguments, pattern), arguments)

    def test_accelerate_cpu_entry_selects_its_own_frontend_cache_and_suite(self):
        tree = ast.parse((ROOT / "noxfile.py").read_text())
        entry = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "benchmark_accelerate"
        )
        entry.decorator_list = []
        with TemporaryDirectory() as directory:
            environment = {"JT_BUILD_NVCC_PATH": "/inherited/nvcc"}
            record = mock.Mock()
            namespace = {
                "os": os,
                "shutil": SimpleNamespace(which=mock.Mock()),
                "_session_env": lambda *args: (Path(directory), environment),
                "_hardware_python": lambda: "benchmark-python",
                "_set_hardware_python_config": mock.Mock(),
                "_record_asv": record,
            }
            exec(
                compile(ast.Module(body=[entry], type_ignores=[]), "noxfile.py", "exec"), namespace
            )
            session = SimpleNamespace(run=mock.Mock(), error=mock.Mock(side_effect=RuntimeError))
            with mock.patch.dict(os.environ, {}, clear=True):
                namespace["benchmark_accelerate"](session)
            self.assertEqual(environment["cache_name"], "asv-accelerate")
            self.assertEqual(environment["JITTOR_TORCH_SHIM"], "1")
            self.assertEqual(environment["JT_BUILD_NVCC_PATH"], "")
            self.assertEqual(environment["nvcc_path"], "")
            self.assertEqual(record.call_args[1]["default_pattern"], r"^accelerate_training\.")
            with mock.patch.dict(os.environ, {"ASV_ACCELERATE_DEVICE": "npu"}, clear=True):
                with self.assertRaises(RuntimeError):
                    namespace["benchmark_accelerate"](session)

    def test_scheduled_ci_reaches_every_general_suite(self):
        cpu = yaml.safe_load((ROOT / ".github/workflows/cpu.yml").read_text())["jobs"]["benchmark"]
        patterns = cpu["env"]
        regular: re.Pattern[str] = re.compile(patterns["ASV_PR_BENCHMARKS"])
        scheduled: re.Pattern[str] = re.compile(patterns["ASV_SCHEDULED_BENCHMARKS"])
        cuda = yaml.safe_load((ROOT / ".github/workflows/cuda.yml").read_text())["jobs"][
            "benchmark-cuda"
        ]
        command = next(step["run"] for step in cuda["steps"] if "--bench" in step.get("run", ""))
        cuda_match = re.search(r"--bench '([^']+)'", command)
        self.assertIsNotNone(cuda_match)
        assert cuda_match is not None
        cuda_pattern: re.Pattern[str] = re.compile(cuda_match.group(1))
        for path in (ROOT / "benchmarks/asv").glob("*.py"):
            if path.stem.startswith("_") or path.stem == "accelerate_training":
                continue  # This suite intentionally requires its own Torch process/cache.
            name = path.stem + ".Example.time_run"
            with self.subTest(module=path.stem):
                self.assertRegex(name, cuda_pattern)
        cpu_suites = {"operators", "optimizer_step", "convolution", "reductions"}
        pr_suites = {"operators", "optimizer_step", "convolution"}
        for module in cpu_suites:
            path = ROOT / "benchmarks" / "asv" / (module + ".py")
            tree = ast.parse(path.read_text())
            constants = {}
            for node in tree.body:
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                if not isinstance(node.targets[0], ast.Name):
                    continue
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    continue

            def parameter_values(node):
                try:
                    return ast.literal_eval(node)
                except (TypeError, ValueError):
                    if isinstance(node, ast.Name) and node.id in constants:
                        return constants[node.id]
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "list"
                        and len(node.args) == 1
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id in constants
                    ):
                        return list(constants[node.args[0].id])
                    raise

            for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
                fields = {
                    node.targets[0].id: node.value
                    for node in cls.body
                    if isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                }
                if "params" not in fields or "param_names" not in fields:
                    continue
                names = ast.literal_eval(fields["param_names"])
                if "device" not in names:
                    continue
                params = fields["params"]
                self.assertIsInstance(params, ast.Tuple)
                assert isinstance(params, ast.Tuple)
                parameters = [parameter_values(item) for item in params.elts]
                for values in itertools.product(*parameters):
                    by_name = dict(zip(names, values))
                    # ASV serializes each parameter with repr, so strings in
                    # real benchmark IDs include quotes: ``('jittor', 'cpu', ...)``.
                    benchmark = "%s.%s.time_run(%s)" % (
                        module,
                        cls.name,
                        ", ".join(map(repr, values)),
                    )
                    eligible = (
                        by_name["device"] == "cpu" and by_name.get("backend", "jittor") == "jittor"
                    )
                    with self.subTest(benchmark=benchmark):
                        if eligible:
                            self.assertRegex(benchmark, scheduled)
                            if module in pr_suites:
                                self.assertRegex(benchmark, regular)
                        else:
                            self.assertNotRegex(benchmark, scheduled)
                            self.assertNotRegex(benchmark, regular)
        reduction_cpu = "reductions.ReductionBenchmarks.time_reduce(jittor, cpu, sum, 1M)"
        self.assertIsNone(regular.search(reduction_cpu))
        self.assertIsNotNone(scheduled.search(reduction_cpu))
