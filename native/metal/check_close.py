"""Exercise the actual tfm_destroy body against a throwing mocked Context.

This bounded C++ control tests ownership and error propagation. It does not
inject a real GPU fault. Both probes use the one existing native build directory.
The exact pre-repair function is included as a failing negative control, so this
works from a fresh checkout without ignored development artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "native/metal/build"
LEGACY_BODY = "int tfm_destroy(void* p){return protect([&]{delete static_cast<Context*>(p);});}"
LEGACY_SOURCE_SHA256 = "67ca9b824189c3c6b6060a543394442e550959ecfd9d99237e2eba3566e16318"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body(path):
    source = path.read_text()
    start = source.index("int tfm_destroy(void* p)")
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            return source[start : index + 1]
    raise ValueError("Unbalanced tfm_destroy definition")


PREFIX = r"""
#include <memory>
#include <stdexcept>
#include <string>
#include <cstdio>
static std::string lastError;
static int syncCalls=0,destructors=0;
struct Context {
    bool fail;
    explicit Context(bool value):fail(value){}
    void sync(){++syncCalls;if(fail)throw std::runtime_error("mock GPU completion error");}
    ~Context(){++destructors;}
};
template<class F>int protect(F f){
    try{f();lastError.clear();return 0;}
    catch(const std::exception& e){lastError=e.what();return -1;}
}
"""
SUFFIX = r"""
int main(){
    int zero=tfm_destroy(nullptr);
    bool null_ok=zero==0&&syncCalls==0&&destructors==0;
    int success=tfm_destroy(new Context(false));
    bool success_ok=success==0&&syncCalls==1&&destructors==1;
    int failure=tfm_destroy(new Context(true));
    bool failure_ok=failure==-1&&syncCalls==2&&destructors==2
        &&lastError=="mock GPU completion error";
    std::printf("{\"null_harmless\":%s,\"success_syncs_and_destroys_once\":%s,"
        "\"error_propagates_and_destroys_once\":%s}\n",
        null_ok?"true":"false",success_ok?"true":"false",failure_ok?"true":"false");
    return null_ok&&success_ok&&failure_ok?0:1;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh receipt directory")
    parser.add_argument("--old-source", type=Path, help="Optional retained pre-repair backend.mm")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    BUILD.mkdir(parents=True, exist_ok=True)
    result_path = output / "cleanup-probe.json"
    results = {}
    for name, source in (
        ("old_negative_control", args.old_source),
        ("repaired", ROOT / "native/metal/backend.mm"),
    ):
        function = body(source) if source is not None else LEGACY_BODY
        code = PREFIX + "\n" + function + "\n" + SUFFIX
        path = BUILD / f"close_probe_{name}.cpp"
        path.write_text(code)
        retained = output / path.name
        retained.write_text(code)
        executable = path.with_suffix("")
        command = ["xcrun", "clang++", "-std=c++17", "-O0", "-g0", str(path), "-o", str(executable)]
        build = subprocess.run(command, capture_output=True, text=True, check=True)
        (output / f"{name}-build.log").write_text(build.stdout + build.stderr)
        result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
        (output / f"{name}.log").write_text(result.stdout + result.stderr)
        results[name] = {
            "returncode": result.returncode,
            "checks": json.loads(result.stdout),
            "native_source_sha256": digest(source) if source is not None else None,
            "function_sha256": hashlib.sha256(function.encode()).hexdigest(),
            "source_mode": "extracted_from_file"
            if source is not None
            else "inline_exact_legacy_body",
            "legacy_full_source_sha256": LEGACY_SOURCE_SHA256 if source is None else None,
            "probe_source_sha256": digest(retained),
            "probe_binary_sha256": digest(executable),
            "compile_command": command,
        }
    passed = (
        results["old_negative_control"]["returncode"] == 1
        and results["repaired"]["returncode"] == 0
    )
    result_path.write_text(
        json.dumps(
            {
                "scope": (
                    "Actual tfm_destroy function body with a mocked Context; "
                    "no real GPU fault injection"
                ),
                "passed": passed,
                "controls": results,
                "driver_sha256": digest(Path(__file__)),
                "host_binary_sha256": (
                    digest(BUILD / "libtransformer_metal.dylib")
                    if (BUILD / "libtransformer_metal.dylib").exists()
                    else None
                ),
                "compiler": subprocess.check_output(["xcrun", "clang++", "--version"], text=True),
            },
            indent=2,
        )
        + "\n"
    )
    print(result_path)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
