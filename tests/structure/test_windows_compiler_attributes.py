"""Keep compiler-specific inline attributes portable in the Windows build."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TestWindowsCompilerAttributes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = (ROOT / "src/core/common.h").read_text(encoding="utf-8")
        cls.node_h = (ROOT / "src/core/node.h").read_text(encoding="utf-8")
        cls.node_cc = (ROOT / "src/core/node.cc").read_text(encoding="utf-8")
        cls.loop_pass = (
            ROOT / "src/codegen/opt/pass/loop_to_func_pass.cc"
        ).read_text(encoding="utf-8")

    def test_shared_annotations_keep_msvc_and_gnu_semantics(self):
        self.assertIn("#define JITTOR_FORCE_INLINE __forceinline", self.common)
        self.assertIn("#define JITTOR_NOINLINE __declspec(noinline)", self.common)
        self.assertIn(
            "#define JITTOR_COLD\n#elif defined(__GNUC__) || defined(__clang__)",
            self.common,
        )
        self.assertIn(
            "#define JITTOR_FORCE_INLINE __attribute__((always_inline))", self.common
        )
        self.assertIn("#define JITTOR_NOINLINE __attribute__((noinline))", self.common)
        self.assertIn("#define JITTOR_COLD __attribute__((cold))", self.common)
        self.assertNotIn("__attribute__", self.node_h)
        self.assertNotIn("__attribute__", self.node_cc)
        self.assertIn("JITTOR_FORCE_INLINE bool own()", self.node_h)
        self.assertIn("JITTOR_FORCE_INLINE bool release()", self.node_h)
        self.assertIn("JITTOR_NOINLINE JITTOR_COLD void report_own()", self.node_h)
        self.assertIn("JITTOR_NOINLINE void grow()", self.node_cc)

    def test_generated_loop_helpers_use_msvc_inline_spelling(self):
        self.assertIn('else if (cc_type=="cl")', self.loop_pass)
        self.assertIn(
            '"#define INLINE_FUNC __forceinline static void "', self.loop_pass
        )
        self.assertIn(
            '"#define INLINE_FUNC __attribute__((always_inline)) static void "',
            self.loop_pass,
        )


if __name__ == "__main__":
    unittest.main()
