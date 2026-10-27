"""Windows gate identity and compiler policy across toolchain discovery paths."""

import ast
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from _helpers.child_process import run_python_child  # noqa: E402

SCRIPT = ROOT / 'tools/release/check_windows_msvc.py'


class WindowsMSVCCheckTests(unittest.TestCase):
    def test_help_does_not_import_or_compile_jittor(self):
        result = run_python_child([str(SCRIPT), '--help'],
                                  without_torch_mode=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--wheel', result.stdout)
        self.assertNotIn('Compiling', result.stdout + result.stderr)

    @unittest.skipIf(os.name == 'nt', 'negative host guard requires a non-Windows host')
    def test_non_windows_cannot_emit_a_passed_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'result.json'
            result = run_python_child(
                [str(SCRIPT), '--wheel', str(root / 'missing.whl'),
                 '--checkout', str(root / 'checkout'), '--commit', 'a' * 40,
                 '--output', str(output)], without_torch_mode=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires a real Windows process', result.stderr)
            self.assertFalse(output.exists())



class WindowsCompilerPolicyTests(unittest.TestCase):
    def _flags(self, platform, compiler, toolchain):
        source = (ROOT / 'python/jittor/build/compiler.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        helpers = ast.parse((ROOT / 'python/jittor/build/utils/compiler_flags.py').read_text(encoding='utf-8'))
        block = next(node for node in tree.body
                     if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Compare)
                     and isinstance(node.test.left, ast.Attribute)
                     and isinstance(node.test.left.value, ast.Name)
                     and node.test.left.value.id == 'os'
                     and node.test.left.attr == 'name'
                     and any(isinstance(child, ast.FunctionDef) and child.name == 'fix_cl_flags'
                             for child in ast.walk(node))
                     and any(getattr(value, 'value', getattr(value, 's', None)) == 'nt'
                             for value in node.test.comparators))
        selected = ast.parse('')
        selected.body = helpers.body + [block]
        utilities = SimpleNamespace(msvc_path=toolchain, get_py3_link_path=lambda: 'C:/Python/libs')
        original = ' -std=c++14 -fPIC -lstdc++ -ldl '
        namespace = dict(os=SimpleNamespace(name=platform), cc_type=compiler,
                         cc_flags=original, jit_utils=utilities, sys=sys)
        with mock.patch.dict(sys.modules, {'jittor_utils': utilities}):
            exec(compile(selected, 'compiler.py', 'exec'), namespace)
        return namespace['cc_flags'], original

    def test_system_and_bundled_msvc_both_disable_windows_minmax_macros(self):
        for toolchain in ('', 'C:/downloaded-msvc'):
            with self.subTest(toolchain=toolchain):
                flags, _ = self._flags('nt', 'cl', toolchain)
                self.assertEqual(flags.split().count('-DNOMINMAX'), 1)
                for required in ('-EHa', '-MD', '-utf-8'):
                    self.assertIn(required, flags.split())

    def test_non_msvc_flags_are_unchanged(self):
        for platform, compiler in (('posix', 'g++'), ('nt', 'g++')):
            with self.subTest(platform=platform, compiler=compiler):
                flags, original = self._flags(platform, compiler, '')
                self.assertEqual(flags, original)


if __name__ == '__main__':
    unittest.main()
