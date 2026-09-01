# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlaceholderRange:
    offset: int = 0
    length: int = 0
    is_embeds: bool = False


@dataclass
class MultiModalFeatureSpec:
    data: Any = None
    modality: str = ""
    identifier: str = ""
    mm_position: PlaceholderRange = field(default_factory=PlaceholderRange)
    mm_hash: str | None = None


# Field/kwargs types are no-op placeholders.
MultiModalKwargsItem = Any
MultiModalKwargsItems = Any
MultiModalKwargsOptionalItems = Any
BatchedTensorInputs = Any
NestedTensors = Any
AudioItem = Any
ImageItem = Any
VideoItem = Any
VisionChunk = Any
VisionChunkImage = Any
VisionChunkVideo = Any
BaseMultiModalField = Any
MultiModalBatchedField = Any
MultiModalFieldConfig = Any
MultiModalFieldElem = Any
MultiModalFlatField = Any
MultiModalSharedField = Any
