# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Safetensors-only model loader for the focused DeepSeek build."""

from __future__ import annotations

import glob
import os
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    safetensors_weights_iterator,
)
from vllm.tracing import instrument

logger = init_logger(__name__)


@dataclass(frozen=True)
class SafetensorsSource:
    """A source for safetensors weights."""

    model_or_path: str
    """The model ID or local path."""

    revision: str | None
    """The optional model revision."""

    prefix: str = ""
    """A prefix to prepend to all weights."""


@dataclass(frozen=True)
class ResolvedSafetensors:
    """Resolved safetensors file locations after discovery/download."""

    directory: Path
    files: tuple[Path, ...]


class SafetensorsModelLoader(BaseModelLoader):
    """Model loader that loads safetensors weights from disk or Hub."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self.local_expert_ids: set[int] | None = None

    def prepare_weights(
        self, model_name_or_path: str, revision: str | None
    ) -> ResolvedSafetensors:
        """Discover or download safetensors weights for the model.

        Args:
            model_name_or_path: Local directory or Hub repository ID.
            revision: Optional revision for Hub download.

        Returns:
            ResolvedSafetensors with the directory and file list.

        Raises:
            RuntimeError: If no safetensors weights are found.
        """
        is_local = os.path.isdir(model_name_or_path)
        allow_patterns = ["*.safetensors"]
        index_file = SAFE_WEIGHTS_INDEX_NAME

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
            )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                break

        if not is_local and len(hf_weights_files) > 1:
            download_safetensors_index_file_from_hf(
                model_name_or_path,
                index_file,
                cache_dir=self.load_config.download_dir,
                revision=revision,
            )
        hf_weights_files = filter_duplicate_safetensors_files(
            hf_weights_files, hf_folder, index_file
        )

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"No safetensors weights found at `{model_name_or_path}`"
            )

        return ResolvedSafetensors(
            directory=Path(hf_folder),
            files=tuple(Path(f) for f in hf_weights_files),
        )

    def _get_weights_iterator(
        self, source: SafetensorsSource
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        resolved = self.prepare_weights(source.model_or_path, source.revision)
        weights_iterator = safetensors_weights_iterator(
            [str(f) for f in resolved.files],
            self.load_config.use_tqdm_on_load,
            self.load_config.safetensors_load_strategy,
            local_expert_ids=self.local_expert_ids,
            safetensors_prefetch_num_threads=(
                self.load_config.safetensors_prefetch_num_threads
            ),
            safetensors_prefetch_block_size=(
                self.load_config.safetensors_prefetch_block_size
            ),
        )
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        primary_weights = SafetensorsSource(
            model_config.model,
            model_config.revision,
            prefix="",
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[SafetensorsSource],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self.prepare_weights(model_config.model, model_config.revision)

    def _init_ep_weight_filter(self, model_config: ModelConfig) -> None:
        """Compute local expert ids for EP weight filtering."""
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config

        if not (
            model_config.is_moe
            and parallel_config.enable_expert_parallel
            and parallel_config.enable_ep_weight_filter
        ):
            return

        num_experts = model_config.get_num_experts()
        if num_experts <= 0:
            return

        from vllm.distributed import (
            get_dp_group,
            get_pcp_group,
            get_tensor_model_parallel_rank,
        )

        dp_size = parallel_config.data_parallel_size
        tp_size = parallel_config.tensor_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        ep_size = dp_size * pcp_size * tp_size
        ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

        self.local_expert_ids = compute_local_expert_ids(
            num_experts,
            ep_size,
            ep_rank,
            placement=parallel_config.expert_placement_strategy,
        )
        if self.local_expert_ids is not None:
            logger.info_once(
                "EP weight filter: ep_size=%d, ep_rank=%d, loading %d/%d experts",
                ep_size,
                ep_rank,
                len(self.local_expert_ids),
                num_experts,
            )

    @instrument(span_name="Load weights")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        self._init_ep_weight_filter(model_config)

        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))

        # Quantized checkpoints legitimately omit parameters that are
        # materialized during processing, so only check unquantized ones.
        if model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                logger.warning(
                    "Following weights were not loaded from checkpoint: %s",
                    weights_not_loaded,
                )

        logger.info_once("Loading weights complete.")

    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        from vllm.model_executor.model_loader.utils import (
            initialize_model,
            process_weights_after_loading,
        )
        from vllm.utils.torch_utils import set_default_torch_dtype

        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            logger.debug("Loading weights on %s ...", load_device)
            self.load_weights(model, model_config)

            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
