# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Model configs may be defined in this directory for the following reasons:

- There is no configuration file defined by HF Hub or Transformers library.
- There is a need to override the existing config to support vLLM.
- The HF model_type isn't recognized by the Transformers library but can
  be mapped to an existing Transformers config, such as
  deepseek-ai/DeepSeek-V3.2-Exp.
"""

from __future__ import annotations

import importlib

_CLASS_TO_MODULE: dict[str, str] = {
    "EAGLEConfig": "vllm.transformers_utils.configs.eagle",
    "SpeculatorsConfig": "vllm.transformers_utils.configs.speculators",
    # Special case: DeepseekV3Config is from HuggingFace Transformers
    "DeepseekV3Config": "transformers",
}

__all__ = [
    "EAGLEConfig",
    "SpeculatorsConfig",
    "DeepseekV3Config",
]


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module = importlib.import_module(_CLASS_TO_MODULE[name])
        return getattr(module, name)

    raise AttributeError(f"module 'configs' has no attribute '{name}'")


def __dir__():
    return sorted(__all__)
