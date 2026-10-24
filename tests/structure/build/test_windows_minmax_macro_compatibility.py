"""Compile the public log header with Windows-style min/max macros enabled."""

import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[3]


SOURCE = r"""
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include "core/types.h"

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#include "utils/log.h"

int main() {
    std::uint8_t byte = 0;
    assert(jittor::parse_env_value(std::string("1"), byte) && byte == 1);
    assert(jittor::parse_env_value(std::string("255"), byte) && byte == 255);
    assert(!jittor::parse_env_value(std::string("256"), byte));

    std::int8_t signed_byte = 0;
    assert(jittor::parse_env_value(std::string("-128"), signed_byte) &&
           signed_byte == -128);
    assert(jittor::parse_env_value(std::string("127"), signed_byte) &&
           signed_byte == 127);
    assert(!jittor::parse_env_value(std::string("-129"), signed_byte));
    assert(!jittor::parse_env_value(std::string("128"), signed_byte));
}
"""


def test_log_header_integer_parser_survives_windows_minmax_macros(tmp_path):
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    assert compiler, "a C++ compiler is required for the log header compatibility gate"

    source = tmp_path / "windows_minmax_probe.cc"
    executable = tmp_path / "windows_minmax_probe"
    source.write_text(SOURCE)
    subprocess.run(
        [compiler, "-std=c++14", "-I", str(ROOT / "src"), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run([str(executable)], check=True, timeout=30)
