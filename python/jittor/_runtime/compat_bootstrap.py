"""Native-owned, stdlib-only entry points for optional compatibility domains."""

import importlib
import os
import sys
from types import SimpleNamespace

from .import_aliases import install_aliases, register_alias_provider


for _prefix in ("jittor.torch_compat", "jittor.torch_fsdp2_compat",
                "jittor.torch_shim", "jittor.triton_shim"):
    register_alias_provider(_prefix, "jittor.compat._aliases")
del _prefix


def is_truthy(value):
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _requested(environ):
    return bool(is_truthy(environ.get("JITTOR_TORCH_SHIM"))
                or environ.get("JITTOR_TORCH_PROJECT_ROOT")
                or environ.get("JITTOR_TORCH_RUNTIME_ROOT"))


def _select_compat_root(environ):
    """Put an explicit compat package root ahead of the installed Jittor tree.

    This runs while ``jittor`` is initializing, before importing any optional
    compatibility module can activate the Torch shim. Keeping the selection
    here lets a process use one exact compat source tree from its first import
    without changing ``JITTOR_TORCH_SHIM`` or replacing the Jittor core path.
    """
    root = environ.get("JITTOR_COMPAT_ROOT")
    if not root:
        return
    if not os.path.isabs(root):
        raise ValueError("JITTOR_COMPAT_ROOT must be an absolute path")
    root = os.path.realpath(os.path.expanduser(root))
    if not os.path.isfile(os.path.join(root, "compat", "__init__.py")):
        raise ImportError(
            "JITTOR_COMPAT_ROOT must contain compat/__init__.py: " + root
        )
    jittor = sys.modules.get("jittor")
    package_path = getattr(jittor, "__path__", None)
    if package_path is None:
        raise RuntimeError("compat root selection requires the initializing jittor package")
    package_path[:] = [root] + [
        entry for entry in package_path if os.path.realpath(entry) != root
    ]


def _compat_module(name, environ=None):
    environ = os.environ if environ is None else environ
    _select_compat_root(environ)
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name in ("jittor.compat", name):
            raise ModuleNotFoundError(
                "Torch compatibility was requested but is not installed; "
                "install the jittor-compat distribution", name=error.name
            ) from error
        raise


def prepare_import_environment(argv=None, environ=None):
    environ = os.environ if environ is None else environ
    _select_compat_root(environ)
    if not _requested(environ):
        return SimpleNamespace(active=False)
    preflight = _compat_module("jittor.compat.shim.preflight", environ=environ)
    return preflight.prepare_import_environment(argv=argv, environ=environ)


def compose(root_module, core_flags, strict=True, preflight=None):
    """Keep plain native startup independent of the optional distribution."""
    if getattr(preflight, "active", False) or _requested(os.environ):
        runtime = _compat_module("jittor.compat.runtime")
        return runtime.compose(root_module, core_flags, strict=strict, preflight=preflight)
    aliases = install_aliases(root_module)
    return SimpleNamespace(torch_reports=(), integrations={}, aliases=aliases)
