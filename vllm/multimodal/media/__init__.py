# SPDX-License-Identifier: Apache-2.0
from typing import Any

MediaConnector = Any


class _MediaConnectorRegistry:
    def load(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def register(self, *args: Any, **kwargs: Any) -> None:
        pass


MEDIA_CONNECTOR_REGISTRY = _MediaConnectorRegistry()
