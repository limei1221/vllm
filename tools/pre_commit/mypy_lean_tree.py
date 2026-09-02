# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Type-check the whole `vllm` package, not just the changed files.

`tools/pre_commit/mypy.py` only sees the files in a commit, so a call site and
the function it calls are checked together only when both happen to change.
That is how `init_fp8_linear_kernel` came to be rewritten without its caller in
`fp8.py`: each file was clean on its own. Checking the package as a whole is
what catches that class of break, and it costs a few seconds.

Files in BASELINE still carry errors from before this check existed. They are
skipped rather than fixed here so the hook can go green now; `follow_imports`
still reads them for types, so only their own diagnostics are suppressed. The
list is meant to shrink -- do not add to it.
"""

import subprocess
import sys
from pathlib import Path

BASELINE = (
    "vllm/model_executor/models/utils.py",
    "vllm/model_executor/models/module_mapping.py",
    "vllm/model_executor/models/interfaces.py",
    "vllm/model_executor/models/interfaces_base.py",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    tracked = subprocess.run(
        ["git", "ls-files", "vllm/*.py", "vllm/**/*.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    targets = [f for f in tracked if f not in BASELINE]
    if not targets:
        return 0

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--python-version",
            sys.argv[1] if len(sys.argv) > 1 else "3.12",
            "--follow-imports",
            "silent",
            "--ignore-missing-imports",
            "--check-untyped-defs",
            *targets,
        ],
        cwd=repo_root,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
