# SPDX-License-Identifier: Apache-2.0


class MultiModalCacheMissError(Exception):
    """Raised only by the (removed) multimodal cache; never raised now."""


class BaseMultiModalProcessorCache:
    """No-op cache stub."""

    def get(self, *args, **kwargs):
        raise MultiModalCacheMissError()

    def put(self, *args, **kwargs) -> None:
        pass
