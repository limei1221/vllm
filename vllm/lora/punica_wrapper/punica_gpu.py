# SPDX-License-Identifier: Apache-2.0
"""No-op stub. PunicaWrapperGPU is removed from the lean build."""

from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase


class PunicaWrapperGPU(PunicaWrapperBase):
    pass
