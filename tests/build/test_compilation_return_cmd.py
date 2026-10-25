"""Return the output-specific command for split compiler builds."""

import importlib.util
import ntpath
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def _load_compilation(monkeypatch):
    """Load the command planner without importing the compiled Jittor runtime."""
    lock_module = ModuleType("jittor_utils")
    lock_module.lock = SimpleNamespace(lock_scope=lambda: lambda fn: fn)

    compiler = ModuleType("test_jittor_build.compiler")
    build_package = ModuleType("test_jittor_build")
    build_package.__path__ = []
    build_package.compiler = compiler

    monkeypatch.setitem(sys.modules, "jittor_utils", lock_module)
    monkeypatch.setitem(sys.modules, "test_jittor_build", build_package)
    monkeypatch.setitem(sys.modules, "test_jittor_build.compiler", compiler)

    source = Path(__file__).resolve().parents[2] / "python/jittor/build/compilation.py"
    spec = importlib.util.spec_from_file_location("test_jittor_build.compilation", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, compiler


def test_return_cmd_is_final_link_command_for_split_build(monkeypatch):
    compilation, compiler = _load_compilation(monkeypatch)
    compile_commands = []
    cache_commands = []

    def record_compile(commands, *_args):
        compile_commands.extend(commands)

    def record_cache(command, *_args):
        cache_commands.append(command)
        return True

    def remove_flags(flags, removed):
        return " ".join(
            token for token in flags.split()
            if not any(token.startswith(prefix) for prefix in removed)
        )

    compiler.os = SimpleNamespace(
        name="nt",
        path=ntpath,
        makedirs=lambda *_args, **_kwargs: None,
    )
    compiler.cache_path = r"C:\cache"
    compiler.jittor_path = r"C:\repo\python\jittor"
    compiler.sys = SimpleNamespace(executable=r"C:\Python\python.exe")
    compiler._source_path = lambda path: path
    compiler.fix_cl_flags = lambda command: command
    compiler.remove_flags = remove_flags
    compiler.lto_flags = ""
    compiler.has_cuda = compiler.has_rocm = 0
    compiler.nvcc_path = ""
    compiler.jit_utils = SimpleNamespace(
        run_cmds=record_compile,
        cc=SimpleNamespace(cache_compile=record_cache),
    )

    args = (
        "cl.exe",
        "-shared -O2",
        [r"C:\repo\src\first.cc", r"C:\repo\src\second.cc"],
        r"C:\cache\jit_utils_core.dll",
        True,
    )
    link_command = compilation.compile(*args, return_cmd=True)

    assert compile_commands == []
    assert cache_commands == []
    assert r"C:\cache\obj_files\first.cc.o" in link_command
    assert r"C:\cache\obj_files\second.cc.o" in link_command
    assert r"C:\cache\jit_utils_core.dll" in link_command

    assert compilation.compile(*args) is True
    assert len(compile_commands) == 2
    assert "dumpdef.py" in cache_commands[0]
    assert cache_commands[-1] == link_command
