# SPDX-License-Identifier: Apache-2.0
"""No-op stubs for ngram GPU proposer."""


class NgramProposerGPU:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("NgramGPU proposer is not supported in the lean build")


def copy_num_valid_draft_tokens(*args, **kwargs) -> None:
    pass


def update_ngram_gpu_tensors_incremental(*args, **kwargs) -> None:
    pass


def update_scheduler_for_invalid_drafts(*args, **kwargs) -> None:
    pass
