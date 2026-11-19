"""Execute actual entry predicates/callbacks without importing native Jittor."""
import ast
from _helpers.paths import REPO_ROOT as ROOT
import importlib
import os
import sys
import types
from types import SimpleNamespace
from unittest import mock


def _functions(path, names, namespace):
    source = ast.parse((ROOT / path).read_text())
    selected = [node for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def test_native_compat_request_uses_explicit_environment_only():
    placeholder = types.SimpleNamespace(_jittor_torch_shim_placeholder=True)
    namespace = {"sys": types.SimpleNamespace(modules={"torch": placeholder})}
    _functions("python/jittor/_runtime/compat_bootstrap.py", {"is_truthy", "_requested"}, namespace)
    requested = namespace["_requested"]
    assert not requested({})
    assert not requested({"JITTOR_TORCH_SHIM": "0"})
    assert requested({"JITTOR_TORCH_SHIM": "1"})
    assert requested({"JITTOR_TORCH_PROJECT_ROOT": "/explicit/project"})
    assert requested({"JITTOR_TORCH_RUNTIME_ROOT": "/explicit/runtime"})


def test_explicit_compat_root_is_selected_before_optional_module_import(tmp_path):
    namespace = {"importlib": importlib, "os": os, "sys": sys}
    _functions("python/jittor/_runtime/compat_bootstrap.py",
               {"_select_compat_root", "_compat_module"}, namespace)
    overlay = tmp_path / "overlay"
    installed = tmp_path / "installed-jittor"
    for source_root, origin in ((overlay, "overlay"), (installed, "installed")):
        compat = source_root / "compat"
        shim = compat / "shim"
        shim.mkdir(parents=True)
        (compat / "__init__.py").write_text("", encoding="utf-8")
        (shim / "__init__.py").write_text("", encoding="utf-8")
        (shim / "preflight.py").write_text(
            "ORIGIN = {!r}\n".format(origin), encoding="utf-8")
    native = types.ModuleType("jittor")
    native.__path__ = [str(installed)]

    with mock.patch.dict(sys.modules):
        for name in list(sys.modules):
            if name == "jittor" or name.startswith("jittor.compat"):
                del sys.modules[name]
        sys.modules["jittor"] = native
        loaded = namespace["_compat_module"](
            "jittor.compat.shim.preflight",
            {"JITTOR_COMPAT_ROOT": str(overlay)},
        )

    assert loaded.ORIGIN == "overlay"
    assert os.path.realpath(loaded.__file__).startswith(os.path.realpath(overlay) + os.sep)
    assert native.__path__ == [str(overlay.resolve()), str(installed)]


def test_explicit_compat_root_fails_closed_when_invalid(tmp_path):
    namespace = {"importlib": importlib, "os": os, "sys": sys}
    _functions("python/jittor/_runtime/compat_bootstrap.py",
               {"_select_compat_root", "_compat_module"}, namespace)
    native = types.ModuleType("jittor")
    native.__path__ = [str(tmp_path / "installed-jittor")]

    with mock.patch.dict(sys.modules, {"jittor": native}):
        with mock.patch.object(importlib, "import_module") as import_module:
            try:
                namespace["_compat_module"](
                    "jittor.compat.shim.preflight",
                    {"JITTOR_COMPAT_ROOT": "relative/path"},
                )
            except ValueError as error:
                assert "absolute path" in str(error)
            else:
                raise AssertionError("relative compat roots must fail closed")
            import_module.assert_not_called()


def test_explicit_compat_root_does_not_enable_the_shim(tmp_path):
    namespace = {"SimpleNamespace": SimpleNamespace, "os": os, "sys": sys}
    _functions("python/jittor/_runtime/compat_bootstrap.py",
               {"is_truthy", "_select_compat_root", "_requested",
                "prepare_import_environment"},
               namespace)
    overlay = tmp_path / "overlay"
    (overlay / "compat").mkdir(parents=True)
    (overlay / "compat" / "__init__.py").write_text("", encoding="utf-8")
    native = types.ModuleType("jittor")
    native.__path__ = [str(tmp_path / "installed-jittor")]

    with mock.patch.dict(sys.modules, {"jittor": native}):
        result = namespace["prepare_import_environment"](
            environ={"JITTOR_COMPAT_ROOT": str(overlay), "JITTOR_TORCH_SHIM": "0"},
        )

    assert not result.active
    assert native.__path__ == [str(overlay.resolve()), str(tmp_path / "installed-jittor")]


def test_historical_alias_requests_independent_activation_without_native_install():
    namespace = {"sys": sys, "__package__": "jittor.compat"}
    _functions("compat/_aliases.py", {"_activate_torch_alias"}, namespace)
    native = types.ModuleType("jittor")
    compatibility = types.SimpleNamespace(install=mock.Mock())
    runtime = types.ModuleType("jittor.compat.shim.runtime")
    runtime.activate = mock.Mock()
    with mock.patch.dict(sys.modules, {"jittor": native, runtime.__name__: runtime}):
        namespace["_activate_torch_alias"](compatibility)
    compatibility.install.assert_not_called()
    runtime.activate.assert_called_once_with(_root_module=native, _composition=True, verbose=False)
