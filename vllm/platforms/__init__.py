# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import traceback
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

from .interface import CpuArchEnum, Platform, PlatformEnum

logger = logging.getLogger(__name__)


def vllm_version_matches_substr(substr: str) -> bool:
    """
    Check to see if the vLLM version matches a substring.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        vllm_version = version("vllm")
    except PackageNotFoundError as e:
        logger.warning(
            "The vLLM package was not found, so its version could not be "
            "inspected. This may cause platform detection to fail."
        )
        raise e
    return substr in vllm_version


def cuda_platform_plugin() -> str | None:
    is_cuda = False
    logger.debug("Checking if CUDA platform is available.")
    try:
        from vllm.utils.import_utils import import_pynvml

        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            # NOTE: Edge case: vllm cpu build on a GPU machine.
            # Third-party pynvml can be imported in cpu build,
            # we need to check if vllm is built with cpu too.
            # Otherwise, vllm will always activate cuda plugin
            # on a GPU machine, even if in a cpu build.
            is_cuda = (
                pynvml.nvmlDeviceGetCount() > 0
                and not vllm_version_matches_substr("cpu")
            )
            if pynvml.nvmlDeviceGetCount() <= 0:
                logger.debug("CUDA platform is not available because no GPU is found.")
            if vllm_version_matches_substr("cpu"):
                logger.debug(
                    "CUDA platform is not available because vLLM is built with CPU."
                )
            if is_cuda:
                logger.debug("Confirmed CUDA platform is available.")
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("Exception happens when checking CUDA platform: %s", str(e))
        if "nvml" not in e.__class__.__name__.lower():
            # If the error is not related to NVML, re-raise it.
            raise e

        # CUDA is supported on Jetson, but NVML may not be.
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") or os.path.exists(
                "/sys/class/tegra-firmware"
            )

        if cuda_is_jetson():
            logger.debug("Confirmed CUDA platform is available on Jetson.")
            is_cuda = True
        else:
            logger.debug("CUDA platform is not available because: %s", str(e))

    return "vllm.platforms.cuda.CudaPlatform" if is_cuda else None


builtin_platform_plugins = {
    "cuda": cuda_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    activated_plugins = []

    for name, func in builtin_platform_plugins.items():
        try:
            assert callable(func)
            if func() is not None:
                activated_plugins.append(name)
        except Exception:
            pass

    if len(activated_plugins) >= 2:
        raise RuntimeError(
            f"Only one platform plugin can be activated, but got: {activated_plugins}"
        )
    if len(activated_plugins) == 1:
        name = activated_plugins[0]
        platform_cls_qualname = builtin_platform_plugins[name]()
        assert platform_cls_qualname is not None
        logger.debug("Automatically detected platform %s.", name)
    else:
        platform_cls_qualname = "vllm.platforms.interface.UnspecifiedPlatform"
        logger.debug("No platform detected, vLLM is running on UnspecifiedPlatform")
    return platform_cls_qualname


_current_platform = None
_init_trace: str = ""

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str):
    if name == "current_platform":
        # lazy init current_platform.
        # 1. out-of-tree platform plugins need `from vllm.platforms import
        #    Platform` so that they can inherit `Platform` class. Therefore,
        #    we cannot resolve `current_platform` during the import of
        #    `vllm.platforms`.
        # 2. when users use out-of-tree platform plugins, they might run
        #    `import vllm`, some vllm internal code might access
        #    `current_platform` during the import, and we need to make sure
        #    `current_platform` is only resolved after the plugins are loaded
        #    (we have tests for this, if any developer violate this, they will
        #    see the test failures).
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


def __setattr__(name: str, value):
    if name == "current_platform":
        global _current_platform
        _current_platform = value
    elif name in globals():
        globals()[name] = value
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = [
    "Platform",
    "PlatformEnum",
    "current_platform",
    "CpuArchEnum",
    "_init_trace",
]
