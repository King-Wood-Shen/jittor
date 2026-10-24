"""Keep the Unix signal symbolizer out of the Windows/MSVC translation unit."""

from pathlib import Path
import unittest


TRACER = Path(__file__).resolve().parents[2] / "src" / "utils" / "tracer.cc"


class TestWindowsTracerPlatformContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TRACER.read_text(encoding="utf-8")

    def test_posix_headers_are_guarded_as_one_platform_group(self):
        expected = "\n".join((
            "#ifndef _WIN32",
            "#include <signal.h>",
            "#include <fcntl.h>",
            "#include <poll.h>",
            "#include <string.h>",
            "#include <sys/wait.h>",
            "#include <unistd.h>",
            "#endif",
            '#include "utils/cross_platform.h"',
        ))
        self.assertIn(expected, self.source)

    def test_forked_symbolizer_is_unix_only(self):
        start = self.source.index("#ifndef _WIN32\n// ---------------------------------------------------------------------------\n// Pre-forked symbolizer")
        end = self.source.index("#endif  // !_WIN32\n\nvoid print_trace()", start)
        unix_symbolizer = self.source[start:end]
        for api in ("fork(", "pipe(", "poll(", "waitpid(", '"/proc/%d/maps"'):
            self.assertIn(api, unix_symbolizer)

    def test_windows_manual_backtrace_path_remains_available(self):
        start = self.source.index("void print_trace()")
        windows_branch = self.source[start:]
        self.assertIn("#ifndef _WIN32", windows_branch)
        self.assertIn("auto cmds = get_cmds(argv);", windows_branch)
        self.assertIn("system_popen(cmds.c_str());", windows_branch)
        self.assertIn("if (!bSuccess)", self.source)


if __name__ == "__main__":
    unittest.main()
