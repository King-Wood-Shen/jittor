"""Pin consistent linkage for helpers compiled into the core extension."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"


def test_new_var_data_owner_declarations_match_its_core_definition():
    declaration = re.compile(
        r"(?m)^\s*(EXTERN_LIB\s+)?PyObject\*\s+new_var_data_owner"
        r"\(VarHolder\*\s+vh\);\s*$"
    )
    binding_header = (SRC / "bindings/pyjt/py_converter.h").read_text(encoding="utf-8")
    core_header = (SRC / "core/var_holder.h").read_text(encoding="utf-8")
    implementation = (SRC / "core/var_holder.cc").read_text(encoding="utf-8")

    binding_match = declaration.search(binding_header)
    core_match = declaration.search(core_header)
    assert binding_match is not None
    assert core_match is not None
    assert binding_match.group(1) is None
    assert core_match.group(1) is None
    assert re.search(
        r"(?m)^PyObject\*\s+new_var_data_owner\(VarHolder\*\s+vh\)\s*\{",
        implementation,
    )
