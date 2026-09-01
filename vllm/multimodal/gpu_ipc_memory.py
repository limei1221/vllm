# SPDX-License-Identifier: Apache-2.0
"""No-op GPU IPC stubs; multimodal GPU IPC memory is removed."""


def reserve_mm_ipc_gpu_memory(*args, **kwargs) -> None:
    pass


def maybe_init_mm_gpu_ipc_pool(*args, **kwargs) -> None:
    pass


def get_mm_gpu_ipc_pool(*args, **kwargs):
    return None
