#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static guard for the lean vLLM tree.

Checks that banned paths, banned terms, and budget limits are not
violated in the focused DeepSeek build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import regex as re

BANNED_PATHS = (
    "vllm/lora",
    "vllm/multimodal",
    "vllm/plugins",
    "vllm/reasoning",
    "vllm/tool_parsers",
    "vllm/v1/pool",
    "vllm/v1/structured_output",
    "vllm/v1/kv_offload",
    "vllm/v1/simple_kv_offload",
    "vllm/model_executor/layers/mamba",
    "vllm/third_party/flash_linear_attention",
)

BANNED_TERMS = {
    "ray": re.compile(r"\bray\b", re.IGNORECASE),
    "external_launcher": "external_launcher",
    "nixl": "nixl",
    "mooncake": "mooncake",
    "lmcache": "lmcache",
    "structured_output": "structured_output",
    "enable_lora": "enable_lora",
    "multimodal": "multimodal",
}

BUDGET_RUNTIME_MAX = 170000
BUDGET_TEST_MAX = 45000


def check_lean_tree(repo_root: Path) -> list[str]:
    """Return a list of violations found in the tree."""
    violations: list[str] = []

    for banned_path in BANNED_PATHS:
        full = repo_root / banned_path
        if full.exists():
            violations.append(f"Banned path exists: {banned_path}")

    for py_file in (repo_root / "vllm").rglob("*.py"):
        rel = py_file.relative_to(repo_root)
        if any(str(rel).startswith(bp) for bp in BANNED_PATHS):
            continue
        try:
            text = py_file.read_text()
        except Exception:
            continue
        for term_name, pattern in BANNED_TERMS.items():
            if isinstance(pattern, re.Pattern):
                found = pattern.search(text)
            else:
                found = pattern in text.lower()
            if found:
                violations.append(f"Banned term '{term_name}' in {rel}")

    return violations


def check_budgets(repo_root: Path) -> list[str]:
    """Check runtime and test LOC budgets."""
    violations: list[str] = []

    def count_lines(directory: Path) -> int:
        total = 0
        for f in directory.rglob("*.py"):
            try:
                total += sum(1 for _ in f.open())
            except Exception:
                pass
        return total

    runtime_loc = count_lines(repo_root / "vllm")
    test_loc = count_lines(repo_root / "tests")

    if runtime_loc > BUDGET_RUNTIME_MAX:
        violations.append(
            f"Runtime LOC {runtime_loc} exceeds budget {BUDGET_RUNTIME_MAX}"
        )
    if test_loc > BUDGET_TEST_MAX:
        violations.append(f"Test LOC {test_loc} exceeds budget {BUDGET_TEST_MAX}")

    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    all_violations: list[str] = []

    if "--all" in sys.argv or "--tree" in sys.argv:
        all_violations.extend(check_lean_tree(repo_root))
    if "--all" in sys.argv or "--budgets" in sys.argv:
        all_violations.extend(check_budgets(repo_root))

    if not all_violations:
        print("Lean tree check passed.")
        return 0

    for v in all_violations:
        print(f"VIOLATION: {v}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
