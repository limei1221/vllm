# SPDX-License-Identifier: Apache-2.0
"""No-op stub. Multimodal video support is removed from the lean build."""


class _VideoLoaderRegistry:
    def backend_requires_gpu(self, *args, **kwargs) -> bool:
        return False


VIDEO_LOADER_REGISTRY = _VideoLoaderRegistry()
