# SPDX-License-Identifier: Apache-2.0
from typing import Any

ProcessorInputs = Any


class BaseMultiModalProcessor:
    """No-op processor stub."""


class EncDecMultiModalProcessor(BaseMultiModalProcessor):
    """No-op encoder-decoder processor stub."""


class TimingContext:
    """No-op context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
