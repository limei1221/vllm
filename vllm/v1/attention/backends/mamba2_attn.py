# SPDX-License-Identifier: Apache-2.0
"""No-op stub. Mamba2 attention is removed from the lean build."""


class Mamba2AttentionMetadataBuilder:
    def __init__(self, *args, **kwargs) -> None:
        pass


class Mamba2AttentionBackend:
    pass
