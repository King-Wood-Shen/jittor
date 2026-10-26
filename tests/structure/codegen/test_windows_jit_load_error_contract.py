"""Keep the Windows JIT symbol failure diagnostic useful and stable."""

from pathlib import Path
import unittest


_SOURCE = Path(__file__).parents[3] / "src" / "codegen" / "jit_compiler.cc"


class TestWindowsJitLoadErrorContract(unittest.TestCase):
    def test_symbol_error_is_captured_before_library_cleanup(self):
        source = _SOURCE.read_text(encoding="utf-8")
        load_start = source.index("jit_op_entry_t load_jit_lib(")
        load_end = source.index("static void invalidate_jit_cache_key", load_start)
        load_source = source[load_start:load_end]

        get_proc = load_source.index("GetProcAddress(")
        capture = load_source.index("GetLastError()", get_proc)
        formatted = load_source.index("windows_error_message(symbol_error)", capture)
        cleanup = load_source.index("FreeLibrary(", formatted)
        self.assertLess(get_proc, capture)
        self.assertLess(capture, formatted)
        self.assertLess(formatted, cleanup)

    def test_formatter_has_system_text_and_numeric_fallback(self):
        source = _SOURCE.read_text(encoding="utf-8")
        start = source.index("static string windows_error_message(DWORD error_code)")
        end = source.index("#endif", start)
        formatter = source[start:end]

        self.assertIn("FormatMessageA(", formatter)
        self.assertIn("FORMAT_MESSAGE_FROM_SYSTEM", formatter)
        self.assertIn('return "Windows error " + std::to_string(error_code)', formatter)
        self.assertIn("LocalFree(message_buffer)", formatter)


if __name__ == "__main__":
    unittest.main()
