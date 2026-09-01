# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stub. Multimodal encoder cache is removed from the lean build."""


class EncoderCache:
    def __init__(self, *args, **kwargs) -> None:
        self.mm_features: dict = {}
        self.encoder_outputs: dict = {}

    def reset(self, *args, **kwargs) -> None:
        pass

    def get(self, key, default=None):
        return self.mm_features.get(key, default)
